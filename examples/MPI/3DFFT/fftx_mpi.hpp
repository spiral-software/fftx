#include "fftx3.hpp"
#include <mpi.h>

enum FFTX_Distribution : int {FFTX_NO_DIST, FFTX_GRID_X, FFTX_GRID_Y, FFTX_GRID_Z};

const int FFTX_CPU = 0;
const int FFTX_GPU = 1;
const int FFTX_HYBRID = 2;

const int FFTX_DEFAULT_LOCAL_LAYOUT = 0;

using namespace fftx;
using namespace std;

vector<string> dagnodes;

template<int DIM, typename T>
struct d_array_t
  {
    d_array_t(const box_t<DIM>& a_box,  const std::vector<FFTX_Distribution>& global_layout, const std::vector<int>local_layout = vector<int>())
      :m_domain(a_box),  m_dist(global_layout)
    {      
      m_array = array_t<DIM, T>(m_domain);
    }    
    
    d_array_t() = default;

    d_array_t(global_ptr<T>&& p, const box_t<DIM>& a_box,const std::vector<FFTX_Distribution>& global_layout)
      :m_domain(a_box), m_dist(global_layout)
    {
      m_array = array_t<DIM, T>(p, m_domain);
    }

    vector<FFTX_Distribution> m_dist;
    vector<int> m_layout;        
    box_t<DIM>  m_domain;
    array_t<DIM, T> m_array;
  };


template<int DIM>
void openScalarDAG(box_t<DIM>& grid)
{

  dagnodes.clear();
  
  cout<<"Load(fftx);"<<endl;
  cout<<"ImportAll(fftx);"<<endl;
  cout<<"Load(mpi);"<<endl;
  cout<<"ImportAll(mpi);"<<endl;
  cout<<endl;
  
  cout<<"pg := [";
  for (int i = 0; i !=grid.extents().dim()-1; ++i){
    cout<<grid.extents()[i]<<",";
  }
  cout<<grid.extents()[DIM-1];
  cout<<"];"<<endl;
  cout<<"procGrid := MPIProcGridND(pg);"<<endl;
  cout<<"conf := LocalConfig.mpi.confMPIGPUDevice(procGrid);"<<endl;

  cout<<"name := \"fftx_default_name\";"<<endl;
  cout<<endl;  
}


//currently hard coded to pencil decomp and no batch
template<int DIM, typename T>
void MDDFT(const point_t<DIM>& extents, int batch,           //extents is the size of the DFT  
	   d_array_t<DIM, T>& destination,
	   d_array_t<DIM, T>& source)
{    
  cout<<"N := [";
  for (int i = 0; i !=source.m_domain.extents().dim()-1; ++i){
    cout<<source.m_domain.extents()[i]<<",";
  }
  cout<<source.m_domain.extents()[DIM-1];  
  cout<<"];"<<endl;
  
  //figure out how data is distributed
  cout<<"#Assumes default layout "<<endl;
  cout<<"Nlocal := N{[1]}::List([2, 3], i-> N[i]/pg[i-1]);"<<endl;


  cout<<"localBrick := TArrayND(TComplex, Nlocal, dimXYZ);"<<endl;

  cout<<"dataLayout := TGlobalArrayND(procGrid, localBrick);"<<endl;
  cout<<"Xglobal := tcast(TPtr(dataLayout), X);"<<endl;
  cout<<"Yglobal := tcast(TPtr(dataLayout), Y);"<<endl;
  
  cout<<"mddft := TRC(MDDFT(N, -1));"<<endl;

  dagnodes.push_back("TDAGNode(mddft, Yglobal, Xglobal)");

  //cout<<endl;
}

template<typename T, int DIM>
void IMDDFT(const point_t<DIM>& extents, int batch,
	    d_array_t<DIM, T>& destination,
	    d_array_t<DIM, T>& source)
{
  cout<<"imddft := TRC(MDDFT(N, 1));"<<endl;

  cout<<"localBrick := TArrayND(TComplex, Nlocal, dimXYZ);"<<endl;
  cout<<"dataLayout := TGlobalArrayND(procGrid, localBrick);"<<endl;

  cout<<"Xglobal := tcast(TPtr(dataLayout), X);"<<endl;
  cout<<"Yglobal := tcast(TPtr(dataLayout), Y);"<<endl;
  
  //  uint64_t dst_id = destination.m_array.id();
  //  uint64_t src_id = source.m_array.id();  
  dagnodes.push_back("TDAGNode(imddft, dest, src)");
}



template<typename T, int DIM, unsigned long COUNT, int GDIM>
void closeScalarDAG(std::array<d_array_t<DIM,T>, COUNT>& localVars, const char* name, box_t<GDIM>& grid)
{

  cout<<"t := TFCall(TDAG("<<endl; 
  for (auto anode: dagnodes)
    {
      cout<<"    ["<<anode<<"]::"<<endl;
    }
  cout<<"     []), "<<endl;
  cout<<"rec(fname := name, params := [ ]));"<<endl;
  
  cout<<"t.params[2].fname := \""<<name<<"\";"<<endl;
  
  cout<<"opts := conf.getOpts(t);"<<endl;
  cout<<"tt := opts.tagIt(t);"<<endl;
  cout<<"c := opts.fftxGen(tt);"<<endl;

  //temporary infrastructure
  cout<<endl;
  cout<<"# Temporary infrastucture for end-to-end demo"<<endl;
  cout<<"init_comm_fn_name := var(\"init_2d_comms\");"<<endl;
  cout<<"lst := [init_comm_fn_name]::pg::N;"<<endl;
  cout<<"init_comm := ApplyFunc(call, lst);"<<endl;
  cout<<"cc := Collect(c, func);"<<endl;
  cout<<"cc[1].cmd := chain(init_comm, cc[1].cmd);"<<endl;
  cout<<endl;
  cout<<"destroy_comm_fn_name := var(\"destroy_2d_comms\");"<<endl;
  cout<<"destroy_comm := ApplyFunc(call, [destroy_comm_fn_name]);"<<endl;
  cout<<"cc[3].cmd := chain(destroy_comm, cc[3].cmd);"<<endl;

  cout<<"PrintTo(t.params[2].fname::\".fftx.source.cu\", opts.prettyPrint(c));"<<endl;

}

template<int DIM>
void kernel(const d_array_t<DIM, double>& symbol,
	    d_array_t<DIM, std::complex<double>>& destination,
	    const d_array_t<DIM, std::complex<double>>& source)
{
  dagnodes.push_back("TDAGNode(Diag(diagTensor()));");
  //std::cout<<"    TDAGNode(Diag(diagTensor(FDataOfs(symvar,"<<symbol.m_domain.size()<<",0),fConst(TReal, 2, 1))), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  
}
