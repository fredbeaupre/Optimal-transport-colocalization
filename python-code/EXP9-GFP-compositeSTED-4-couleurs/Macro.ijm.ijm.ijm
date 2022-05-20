selectWindow("9-GFP-LAM18-GFP-TAUS460L-BCaMKIIA594-PH635_cs1n1_CompositeSTED.tiff");
Stack.setChannel(1)
run("biop-BrightPink");
Stack.setChannel(2)
run("biop-Azure");
Stack.setChannel(3)
run("biop-SpringGreen");
Stack.setChannel(4)
run("BOP orange");