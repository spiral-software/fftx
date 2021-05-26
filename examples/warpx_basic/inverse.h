// make



using namespace fftx;


tracing=true;
  
box_t<3> domain3({{1,1,1}},{{l,m,n}});
box_t<3> fdomain3({{1,1,1}},{{l/2+1,m,n}});

std::array<array_t<3,std::complex<double>>,2> intermediates {fdomain3, fdomain3};
array_t<3,std::complex<double>> inputs(fdomain3);
array_t<3,double> outputs(domain3);
array_t<3,double> symbol(domain3);


setInputs(inputs);
setOutputs(outputs);

openScalarDAG();
IPRDFT(domain3.extents().flipped(), outputs, inputs);
closeScalarDAG(intermediates, name);
