"""Tests for the reusable helpers in ``deep_sdf.training_latent_field``.

The ``train`` entry point in that module needs a dataset and a full training run,
so it is exercised only by the training tests. Everything around it -- the loss,
the learning-rate schedules, the spline/latent-field construction and the
checkpoint round trips -- is pure and covered here.
"""

import math
import os
import random

import numpy as np
import pytest
import torch

import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.training_latent_field import (
    ClampedL1Loss,
    ConstantLearningRateSchedule,
    CosineAnnealingLRSchedule,
    LearningRateSchedule,
    StepLearningRateSchedule,
    WarmupLearningRateSchedule,
    _make_lr_schedule,
    _seed_worker,
    append_parameter_magnitudes,
    build_template_spline,
    clip_logs,
    get_mean_spline_param_magnitude,
    get_spec_with_default,
    load_latent_fields,
    load_logs,
    load_optimizer,
    make_latent_fields,
    save_latent_fields,
    save_logs,
    save_model,
    save_optimizer,
)

BOUNDS = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


@pytest.fixture(autouse=True)
def _float32():
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(saved)


def test_clamped_l1_loss_clamps_both_sides():
    loss = ClampedL1Loss(clamp_val=0.1)

    # Both values are clamped to +0.1, so the outlier magnitude is discarded.
    assert loss(torch.tensor([5.0]), torch.tensor([0.2])).item() == pytest.approx(0.0)
    # Input clamps to 0.1, target stays at 0.0.
    assert loss(torch.tensor([1.0]), torch.tensor([0.0])).item() == pytest.approx(0.1)
    # Inside the clamp range it is a plain L1 loss.
    assert loss(torch.tensor([0.05]), torch.tensor([-0.05])).item() == pytest.approx(
        0.1
    )
    # Clamping is symmetric about zero.
    assert loss(torch.tensor([-5.0]), torch.tensor([-0.2])).item() == pytest.approx(0.0)


def test_clamped_l1_loss_averages_over_elements():
    loss = ClampedL1Loss(clamp_val=1.0)
    value = loss(torch.tensor([0.0, 0.4]), torch.tensor([0.0, 0.0]))
    assert value.item() == pytest.approx(0.2)


def test_base_learning_rate_schedule_is_abstract():
    with pytest.raises(NotImplementedError):
        LearningRateSchedule().get_learning_rate(0)


def test_constant_schedule_ignores_epoch():
    schedule = ConstantLearningRateSchedule(1e-3)
    assert schedule.get_learning_rate(0) == pytest.approx(1e-3)
    assert schedule.get_learning_rate(10_000) == pytest.approx(1e-3)


def test_step_schedule_decays_on_interval_boundaries():
    schedule = StepLearningRateSchedule(initial=1.0, interval=10, factor=0.5)

    assert schedule.get_learning_rate(0) == pytest.approx(1.0)
    assert schedule.get_learning_rate(9) == pytest.approx(1.0)
    # First decay lands exactly on the interval boundary.
    assert schedule.get_learning_rate(10) == pytest.approx(0.5)
    assert schedule.get_learning_rate(19) == pytest.approx(0.5)
    assert schedule.get_learning_rate(20) == pytest.approx(0.25)


def test_warmup_schedule_ramps_then_holds():
    schedule = WarmupLearningRateSchedule(initial=0.0, warmed_up=1.0, length=10)

    assert schedule.get_learning_rate(0) == pytest.approx(0.0)
    assert schedule.get_learning_rate(5) == pytest.approx(0.5)
    assert schedule.get_learning_rate(10) == pytest.approx(1.0)
    # Past the warmup length the rate is pinned to the warmed-up value.
    assert schedule.get_learning_rate(11) == pytest.approx(1.0)
    assert schedule.get_learning_rate(1000) == pytest.approx(1.0)


def test_cosine_schedule_anneals_from_initial_to_final():
    schedule = CosineAnnealingLRSchedule(initial=1.0, final=0.0, total_epochs=10)

    assert schedule.get_learning_rate(0) == pytest.approx(1.0)
    # cos(pi/2) == 0, so the midpoint sits halfway between initial and final.
    assert schedule.get_learning_rate(5) == pytest.approx(0.5)
    assert schedule.get_learning_rate(10) == pytest.approx(0.0)
    # Beyond total_epochs the schedule clamps instead of turning back up.
    assert schedule.get_learning_rate(11) == pytest.approx(0.0)

    # Monotonically decreasing over the annealing window.
    rates = [schedule.get_learning_rate(e) for e in range(11)]
    assert rates == sorted(rates, reverse=True)


def test_cosine_schedule_respects_nonzero_final():
    schedule = CosineAnnealingLRSchedule(initial=1.0, final=0.2, total_epochs=4)
    assert schedule.get_learning_rate(0) == pytest.approx(1.0)
    assert schedule.get_learning_rate(2) == pytest.approx(0.6)
    assert schedule.get_learning_rate(4) == pytest.approx(0.2)
    expected = 0.2 + 0.5 * 0.8 * (1 + math.cos(math.pi * 1 / 4))
    assert schedule.get_learning_rate(1) == pytest.approx(expected)


def test_get_spec_with_default():
    specs = {"present": 1, "falsy": 0, "none": None}

    assert get_spec_with_default(specs, "present", 99) == 1
    assert get_spec_with_default(specs, "missing", 99) == 99
    # Membership, not truthiness: a stored 0 or None must not fall back.
    assert get_spec_with_default(specs, "falsy", 99) == 0
    assert get_spec_with_default(specs, "none", 99) is None


@pytest.mark.parametrize("value", [1e-3, 5, 0])
def test_make_lr_schedule_accepts_bare_numbers(value):
    schedule = _make_lr_schedule(value)
    assert isinstance(schedule, ConstantLearningRateSchedule)
    assert schedule.get_learning_rate(3) == pytest.approx(float(value))


def test_make_lr_schedule_dispatches_on_type():
    constant = _make_lr_schedule({"Type": "Constant", "Value": 0.5})
    assert isinstance(constant, ConstantLearningRateSchedule)

    step = _make_lr_schedule(
        {"Type": "Step", "Initial": 1.0, "Interval": 2, "Factor": 0.1}
    )
    assert isinstance(step, StepLearningRateSchedule)
    assert step.get_learning_rate(2) == pytest.approx(0.1)

    warmup = _make_lr_schedule(
        {"Type": "Warmup", "Initial": 0.0, "WarmedUp": 1.0, "Length": 4}
    )
    assert isinstance(warmup, WarmupLearningRateSchedule)
    assert warmup.get_learning_rate(2) == pytest.approx(0.5)

    cosine = _make_lr_schedule(
        {"Type": "Cosine", "Initial": 1.0, "Final": 0.0, "TotalEpochs": 8}
    )
    assert isinstance(cosine, CosineAnnealingLRSchedule)
    assert cosine.get_learning_rate(8) == pytest.approx(0.0)


def test_make_lr_schedule_type_is_case_insensitive_and_defaults_to_constant():
    assert isinstance(
        _make_lr_schedule(
            {"Type": "sTeP", "Initial": 1.0, "Interval": 1, "Factor": 1.0}
        ),
        StepLearningRateSchedule,
    )
    # No "Type" key at all falls back to Constant.
    assert isinstance(_make_lr_schedule({"Value": 0.25}), ConstantLearningRateSchedule)


def test_make_lr_schedule_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown LR schedule type"):
        _make_lr_schedule({"Type": "does_not_exist"})


def test_clip_logs_truncates_to_epoch():
    # 3 iterations per epoch over 4 epochs.
    loss_log = list(range(12))
    lr_log = [0.1, 0.2, 0.3, 0.4]
    timing_log = [1.0, 2.0, 3.0, 4.0]
    lat_mag_log = [5.0, 6.0, 7.0, 8.0]
    param_mag_log = {"lin.weight": [1, 2, 3, 4], "lin.bias": [9, 8, 7, 6]}

    loss, lr, timing, lat, params = clip_logs(
        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch=2
    )

    assert loss == list(range(6))  # 3 iters/epoch * 2 epochs
    assert lr == [0.1, 0.2]
    assert timing == [1.0, 2.0]
    assert lat == [5.0, 6.0]
    assert params == {"lin.weight": [1, 2], "lin.bias": [9, 8]}


def test_clip_logs_is_a_no_op_at_full_length():
    loss, lr, _, _, _ = clip_logs(
        list(range(4)), [0.1, 0.2, 0.3, 0.4], [0] * 4, [0] * 4, {}, epoch=4
    )
    assert loss == list(range(4))
    assert lr == [0.1, 0.2, 0.3, 0.4]


def test_load_logs_round_trip(tmp_path):
    payload = {
        "epoch": 7,
        "loss": [1.0, 0.5],
        "learning_rate": [0.1],
        "timing": [12.0],
        "latent_magnitude": [3.0],
        "param_magnitude": {"w": [1.0]},
    }
    torch.save(payload, os.path.join(tmp_path, ws.logs_filename))

    loss, lr, timing, lat_mag, param_mag, epoch = load_logs(tmp_path)

    assert loss == payload["loss"]
    assert lr == payload["learning_rate"]
    assert timing == payload["timing"]
    assert lat_mag == payload["latent_magnitude"]
    assert param_mag == payload["param_magnitude"]
    assert epoch == 7


def test_load_logs_raises_for_missing_file(tmp_path):
    with pytest.raises(Exception, match="does not exist"):
        load_logs(tmp_path)


def test_save_logs_writes_checkpoint_and_plot(tmp_path):
    # plot_logs smooths the loss with a width-41 running mean and plots one
    # learning rate per parameter group, so the logs need that much shape.
    epochs = 10
    # Plain floats, as ``train`` logs them via ``loss.item()``: the logs are
    # reloaded with a bare ``torch.load``, which since torch 2.6 defaults to
    # weights_only=True and rejects pickled numpy scalars.
    loss_log = np.linspace(1.0, 0.1, 100).tolist()
    lr_log = [[0.1, 0.01] for _ in range(epochs)]
    timing_log = [1.5] * epochs
    lat_mag_log = [0.3] * epochs
    param_mag_log = {"lin.weight": [1.0] * epochs}

    save_logs(
        tmp_path, loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epochs
    )

    loss, lr, timing, lat_mag, param_mag, epoch = load_logs(tmp_path)
    assert epoch == epochs
    assert loss == loss_log
    assert lr == lr_log
    assert timing == timing_log
    assert lat_mag == lat_mag_log
    assert param_mag == param_mag_log

    # The loss curve is rendered alongside the raw log.
    plot_path = tmp_path / ws.logplot_filename
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_seed_worker_is_deterministic_per_torch_seed():
    def draw():
        _seed_worker(worker_id=0)
        return np.random.rand(3).tolist(), random.random()

    torch.manual_seed(1234)
    first = draw()
    torch.manual_seed(1234)
    second = draw()
    # Same torch seed -> same numpy/random streams in the worker.
    assert first == second

    torch.manual_seed(4321)
    third = draw()
    assert third != first


@pytest.mark.parametrize(
    "tiling, expected_control_points",
    [([1, 1, 1], 8), ([2, 2, 2], 27), ([1, 2, 3], 24), ([4, 1, 1], 20)],
)
def test_build_template_spline_control_point_grid(tiling, expected_control_points):
    """For linear degrees the grid must be prod(tiling_i + 1) control points."""
    spline = build_template_spline(latent_dim=4, tiling=tiling, bounds=BOUNDS)

    control_points = spline.control_points
    assert control_points.shape == (expected_control_points, 4)
    assert list(spline.degrees) == [1, 1, 1]
    # Control points start at zero so training begins from a neutral field.
    assert (control_points == 0.0).all()


def test_build_template_spline_spans_requested_bounds():
    bounds = torch.tensor([[-2.0, 0.0, 1.0], [3.0, 0.5, 4.0]])
    spline = build_template_spline(latent_dim=2, tiling=[2, 1, 1], bounds=bounds)

    for i_dim, (lo, hi) in enumerate(zip(bounds[0], bounds[1])):
        knots = spline.knot_vectors[i_dim]
        assert min(knots) == pytest.approx(lo.item())
        assert max(knots) == pytest.approx(hi.item())


def test_build_template_spline_honours_explicit_degrees():
    spline = build_template_spline(
        latent_dim=3, tiling=[1, 1, 1], bounds=BOUNDS, degrees=[2, 1, 1]
    )
    assert list(spline.degrees) == [2, 1, 1]
    # prod(p + 1) = 3 * 2 * 2
    assert spline.control_points.shape == (12, 3)


def test_build_template_spline_rejects_mismatched_degrees():
    with pytest.raises(ValueError, match="degrees must have length 3"):
        build_template_spline(
            latent_dim=2, tiling=[1, 1, 1], bounds=BOUNDS, degrees=[1, 1]
        )


def test_make_latent_fields_creates_independent_fields():
    latent_dim, num_scenes = 4, 3
    fields = make_latent_fields(
        num_scenes, latent_dim, [2, 2, 2], BOUNDS, device="cpu", init_std=0.01
    )

    assert len(fields) == num_scenes
    for field in fields:
        params = list(field.parameters())
        assert len(params) == 1
        assert params[0].shape == (27, latent_dim)
        assert params[0].requires_grad

    # Independently initialized, so no two scenes share a parameter tensor.
    first = list(fields[0].parameters())[0]
    second = list(fields[1].parameters())[0]
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)


def test_make_latent_fields_respects_init_std():
    wide = make_latent_fields(4, 8, [2, 2, 2], BOUNDS, device="cpu", init_std=1.0)
    narrow = make_latent_fields(4, 8, [2, 2, 2], BOUNDS, device="cpu", init_std=1e-4)

    def flat_std(fields):
        values = torch.cat(
            [p.detach().flatten() for f in fields for p in f.parameters()]
        )
        return values.std().item()

    wide_std = flat_std(wide)
    narrow_std = flat_std(narrow)

    assert wide_std > narrow_std
    assert wide_std == pytest.approx(1.0, rel=0.2)
    assert narrow_std == pytest.approx(1e-4, rel=0.2)


def test_get_mean_spline_param_magnitude():
    fields = make_latent_fields(2, 4, [1, 1, 1], BOUNDS, device="cpu", init_std=0.01)

    # Set both fields to a known norm: 8x4 entries of 0.5 -> norm sqrt(32)*0.5.
    for field in fields:
        for param in field.parameters():
            param.data.fill_(0.5)
    expected = math.sqrt(8 * 4) * 0.5

    assert get_mean_spline_param_magnitude(fields) == pytest.approx(expected)


def test_get_mean_spline_param_magnitude_of_empty_is_zero():
    assert get_mean_spline_param_magnitude([]) == 0.0
    assert get_mean_spline_param_magnitude(torch.nn.ModuleList()) == 0.0


def test_append_parameter_magnitudes_accumulates_per_parameter():
    model = torch.nn.Linear(3, 1, bias=True)
    torch.nn.init.constant_(model.weight, 0.0)
    torch.nn.init.constant_(model.bias, 0.0)
    log = {}

    append_parameter_magnitudes(log, model)
    assert set(log) == {"weight", "bias"}
    assert log["weight"] == [0.0]

    # A second call appends rather than replacing.
    torch.nn.init.constant_(model.weight, 1.0)  # norm of three ones
    append_parameter_magnitudes(log, model)
    assert log["weight"] == [pytest.approx(0.0), pytest.approx(math.sqrt(3))]
    assert len(log["bias"]) == 2


def test_append_parameter_magnitudes_strips_dataparallel_prefix():
    inner = torch.nn.Linear(2, 1)
    wrapped = torch.nn.ModuleDict({"module": inner})
    log = {}

    append_parameter_magnitudes(log, wrapped)

    # "module.weight" is recorded as "weight" so logs survive un/wrapping.
    assert set(log) == {"weight", "bias"}


def test_save_model_writes_checkpoint(tmp_path):
    decoder = torch.nn.Linear(4, 1)
    save_model(tmp_path, "test.pth", decoder, epoch=3)

    path = os.path.join(ws.get_model_params_dir(tmp_path), "test.pth")
    saved = torch.load(path, weights_only=False)
    assert saved["epoch"] == 3
    assert set(saved["model_state_dict"]) == set(decoder.state_dict())


def test_optimizer_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.123)
    save_optimizer(tmp_path, "opt.pth", optimizer, epoch=5)

    restored = torch.optim.Adam(model.parameters(), lr=0.999)
    epoch = load_optimizer(tmp_path, "opt.pth", restored)

    assert epoch == 5
    assert restored.param_groups[0]["lr"] == pytest.approx(0.123)


def test_load_optimizer_returns_zero_when_missing(tmp_path):
    optimizer = torch.optim.Adam(torch.nn.Linear(2, 1).parameters())
    assert load_optimizer(tmp_path, "absent.pth", optimizer) == 0


def test_latent_fields_checkpoint_round_trip(tmp_path):
    latent_dim, num_scenes = 4, 2
    saved_fields = make_latent_fields(
        num_scenes, latent_dim, [2, 2, 2], BOUNDS, device="cpu", init_std=0.5
    )
    save_latent_fields(
        tmp_path,
        "latents.pth",
        saved_fields,
        epoch=9,
        num_scenes=num_scenes,
        latent_dim=latent_dim,
        device="cpu",
    )

    restored_fields = make_latent_fields(
        num_scenes, latent_dim, [2, 2, 2], BOUNDS, device="cpu", init_std=0.5
    )
    epoch = load_latent_fields(tmp_path, "latents.pth", restored_fields, "cpu")

    assert epoch == 9
    for saved, restored in zip(saved_fields, restored_fields):
        for p_saved, p_restored in zip(saved.parameters(), restored.parameters()):
            torch.testing.assert_close(p_saved, p_restored)


def test_save_latent_fields_writes_dummy_latent_codes(tmp_path):
    """``load_latent_vectors``/``get_model`` expect a latent_codes entry."""
    fields = make_latent_fields(3, 5, [1, 1, 1], BOUNDS, device="cpu")
    save_latent_fields(
        tmp_path,
        "latents.pth",
        fields,
        epoch=1,
        num_scenes=3,
        latent_dim=5,
        device="cpu",
    )

    path = os.path.join(ws.get_latent_codes_dir(tmp_path), "latents.pth")
    data = torch.load(path, weights_only=False)
    assert data["latent_codes"].shape == (3, 5)
    assert torch.count_nonzero(data["latent_codes"]) == 0


def test_load_latent_fields_returns_zero_when_missing(tmp_path):
    fields = make_latent_fields(1, 2, [1, 1, 1], BOUNDS, device="cpu")
    assert load_latent_fields(tmp_path, "absent.pth", fields, "cpu") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
