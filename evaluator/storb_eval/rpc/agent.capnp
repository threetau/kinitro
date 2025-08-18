@0x893bac407c81b48c;

interface Agent {

    struct Tensor {
        data @0   :Data; # tensor bytes tensor.numpy().tobytes()
        shape @1  :List(UInt64); # tensor shape list(tensor.shape())
        dtype @2  :Text; # data type name tensor.dtype()
    }

    act @0 (obs :Data) -> (action :Tensor);
    reset @1 () -> ();
}
