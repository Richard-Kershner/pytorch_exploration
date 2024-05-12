# pytorch_exploration
 Exploring neural networks in pytorch
 
 ## Custom activation
 ### act_bell_relu.py
 pytoch_exploration_test/
     test_act_bell_relu.ipynb
	 test_act_bell_relu.html
 
 ## Autobuild for easily testing mulitple neural network level(s) configurations
 ### model_autobuild.py
 example of dictionary:
 init_layers = {
      0 : {"outCnt": 3, "activation":"input"} # inputLayer... 0 denotes if multiple inputs, first
    ,"in2" : {"outCnt": 3, "activation":"input"}
    ,"full_input" : {"input":[0,"in2"], "activation":"concat"}
    , "A-dense" : {"input":"full_input", "outCnt":9, "activation":"dense"}
    , "A-bell_reLu" : {"input":"full_input", "activation":Bell_reLU()}
    , "A-reLu" : {"input":"full_input", "activation":nn.ReLU()}
    , "B-concat_A" : {"input":["A-dense","A-bell_reLu", "A-reLu"], "activation":"concat"}
    , "B-dense": {"input":"B-concat_A", "outCnt":9, "activation":"dense"}
    , "C-out_dense": {"input":"B-dense", "outCnt":9, "activation":"dense"}
}

Currently call seperately:
    init_layers = reformat_init_layers(init_layers)
	model = ModelFromInitLayers(init_layers, input_layers, output_layers)
	
 pytoch_exploration_test/
     test_model_autobuild.ipynb
	 test_model_autobuild.html