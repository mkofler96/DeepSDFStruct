import torch
import DeepSDFStruct.deep_sdf.workspace as ws


def main(experiment_directory, checkpoint):
    device = torch.device("cpu")
    decoder = ws.load_trained_model(experiment_directory, checkpoint)
    latent = ws.load_latent_vectors(experiment_directory, checkpoint).to(device)
    decoder.eval().to(device)

    example_input = torch.cat([latent[7], torch.tensor([0, 0, 0])]).unsqueeze(0)

    print("Example input1: ", example_input)
    print("Example Output1: ", decoder(example_input))

    example_input = torch.cat([latent[18], torch.tensor([0, 0, 0])]).unsqueeze(0)

    print("Example input2: ", example_input)
    print("Example Output2: ", decoder(example_input))
    decoder_traced = torch.jit.trace(decoder, example_input)
    sm = torch.jit.script(decoder_traced)
    # traced_script_module = torch.jit.trace(decoder, example_input)
    # traced_script_module.save(experiment_directory + "/model.pt")
    sm.save(experiment_directory + "/cpp_model.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_directory", "-e", type=str)
    parser.add_argument("--checkpoint", "-c", type=str)

    args = parser.parse_args()

    main(args.experiment_directory, args.checkpoint)
