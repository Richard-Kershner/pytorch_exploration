import torch


def reformat_init_layers(init_layers):
    dtype = float
    still_run = True
    while still_run:
        still_run = False  # this makes this a do/while...  if the input isn't found...
        for key, i_lay in init_layers.items():
            if i_lay["activation"] == "input":
                i_lay["tensor"] = torch.zeros(i_lay["outCnt"])
            if ("outCnt" not in i_lay) and ("input" in i_lay):
                if isinstance(i_lay["input"], list):  # ---- concat
                    outCnt = 0
                    for pnt in i_lay["input"]:
                        if "outCnt" not in init_layers[pnt]:
                            still_run = True
                            break
                        else:
                            outCnt += init_layers[pnt]["outCnt"]
                    if not still_run:
                        i_lay["outCnt"] = outCnt
                        i_lay["tensor"] = torch.zeros(outCnt)
                else:  # ---- activation not dense
                    if "outCnt" in init_layers[i_lay["input"]]:
                        i_lay["outCnt"] = init_layers[i_lay["input"]]["outCnt"]
                        i_lay["tensor"] = torch.zeros(i_lay["outCnt"])
                    else:
                        still_run = True
                        break
            if i_lay["activation"] == "dense":  # ---- dense
                pnt = i_lay["input"]
                if ("outCnt" in init_layers[pnt]) and ("outCnt" in i_lay):
                    cntIn = init_layers[pnt]['outCnt']
                    cntOut = i_lay["outCnt"]
                    i_lay["activation"] = torch.nn.Linear(cntIn, cntOut, dtype=dtype)
                    i_lay["tensor"] = torch.zeros(i_lay["outCnt"])
                else:
                    still_run = True
    return init_layers

class ModelFromInitLayers(torch.nn.Module):
    def __init__(self, init_layers, input_layers, output_layers):
        super(ModelFromInitLayers, self).__init__()
        self.layers = init_layers
        self.input_layers = input_layers
        self.output_layers = output_layers

    def forward(self, data):
        for i in range(len(data)):
            print("===============", data[i])
            self.layers[self.input_layers[i]]["tensor"] = torch.from_numpy(data[i])

        # ----- Loop through layers and "activate -----
        for key, lay in self.layers.items():
            if lay["activation"] == "input": # nothing for input
                pass  # do nothing

            elif lay["activation"] == "concat": # --- concatenation of layers
                cat_this = []
                for ten_pnt in lay["input"]:
                    cat_this.append(self.layers[ten_pnt]["tensor"])
                lay["tensor"] = torch.concatenate(cat_this)

            else:  # -- all else are activations like reLu, etc.
                source = lay["input"]
                lay["tensor"] = lay["activation"](self.layers[source]["tensor"])
        data_out = []
        for o in self.output_layers:
            data_out.append(self.layers[o]["tensor"])

        return data_out
