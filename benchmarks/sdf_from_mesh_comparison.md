Mesh            | Points     | Backend    | Time (s)  
-------------------------------------------------------
Sphere (Low)    | 10000      | igl        | 0.0124
Sphere (Low)    | 10000      | trimesh    | 0.0155
Sphere (Low)    | 100000     | igl        | 0.0253
Sphere (Low)    | 100000     | trimesh    | 0.0732
Sphere (Low)    | 1000000    | igl        | 0.1554
Sphere (Low)    | 1000000    | trimesh    | 0.7333
Sphere (Med)    | 10000      | igl        | 0.0472
Sphere (Med)    | 10000      | trimesh    | 0.0399
Sphere (Med)    | 100000     | igl        | 0.0889
Sphere (Med)    | 100000     | trimesh    | 0.1181
Sphere (Med)    | 1000000    | igl        | 0.3624
Sphere (Med)    | 1000000    | trimesh    | 0.8615
Bunny (High)    | 10000      | igl        | 0.8877
Bunny (High)    | 10000      | trimesh    | 0.4823
Bunny (High)    | 100000     | igl        | 0.9768
Bunny (High)    | 100000     | trimesh    | 0.6239
Bunny (High)    | 1000000    | igl        | 1.6876
Bunny (High)    | 1000000    | trimesh    | 1.5953

Speedup (IGL vs Trimesh):
Backend                    igl   trimesh  Speedup (x)
Mesh         Points                                  
Bunny (High) 10000    0.887726  0.482313     0.543312
             100000   0.976751  0.623863     0.638712
             1000000  1.687581  1.595282     0.945307
Sphere (Low) 10000    0.012439  0.015542     1.249462
             100000   0.025271  0.073200     2.896629
             1000000  0.155364  0.733257     4.719598
Sphere (Med) 10000    0.047250  0.039910     0.844664
             100000   0.088914  0.118088     1.328118
             1000000  0.362449  0.861516     2.376932