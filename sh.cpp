#include <iostream> 
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream> 
#include <string> 
#include <vector> 
using namespace std; 

vector<double>label; 
vector<vector<double>>image; 
vector<double>labels; 
vector<vector<double>>images; 

 int ma=240;//隐含层节点个数




int ReverseInt(int i) 
{ 
  unsigned char ch1, ch2, ch3, ch4; 
  ch1 = i & 255; 
  ch2 = (i >> 8) & 255; 
  ch3 = (i >> 16) & 255; 
  ch4 = (i >> 24) & 255; 
  return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4; 
} 



//读取手写字体的label
void read_Mnist_Label(string filename, vector<double>&labels) 
{ 
  ifstream file(filename, ios::binary); 
  if (file.is_open()) 
  { 
    int magic_number = 0; 
    int number_of_images = 0; 
    file.read((char*)&magic_number, sizeof(magic_number)); 
    file.read((char*)&number_of_images, sizeof(number_of_images)); 
    magic_number = ReverseInt(magic_number); 
    number_of_images = ReverseInt(number_of_images); 
    cout << "magic number = " << magic_number << endl; 
    cout << "number of images = " << number_of_images << endl; 
    for (int i = 0; i < number_of_images; i++) 
    { 
      unsigned char label = 0; 
      file.read((char*)&label, sizeof(label)); 
      labels.push_back((double)label); 
    } 
  } 
} 



//读取手写字体文件
void read_Mnist_Images(string filename, vector<vector<double>>&images) 
{ 
  ifstream file(filename, ios::binary); 
  if (file.is_open()) 
  { 
    int magic_number = 0; 
    int number_of_images = 0; 
    int n_rows = 0; 
    int n_cols = 0; 
    unsigned char label; 
    file.read((char*)&magic_number, sizeof(magic_number)); 
    file.read((char*)&number_of_images, sizeof(number_of_images)); 
    file.read((char*)&n_rows, sizeof(n_rows)); 
    file.read((char*)&n_cols, sizeof(n_cols)); 
    magic_number = ReverseInt(magic_number); 
    number_of_images = ReverseInt(number_of_images); 
    n_rows = ReverseInt(n_rows); 
    n_cols = ReverseInt(n_cols); 
    cout << "magic number = " << magic_number << endl; 
    cout << "number of images = " << number_of_images << endl; 
    cout << "rows = " << n_rows << endl; 
    cout << "cols = " << n_cols << endl; 
    for (int i = 0; i < number_of_images; i++) 
    { 
      vector<double>tp; 
      for (int r = 0; r < n_rows; r++) 
      { 
        for (int c = 0; c < n_cols; c++) 
        { 
          unsigned char image = 0; 
          file.read((char*)&image, sizeof(image)); 
          tp.push_back(image); 
        } 
      } 
      images.push_back(tp); 
    } 
  } 
} 


double sigmoid(double a)//S型激活函数
{
  double u=exp(-a);
  return 1.0/(1.0+u);
}


int main() 
{ 

  read_Mnist_Label("train-labels.idx1-ubyte", labels);//读取训练集label 
  /* for (auto iter = labels.begin(); iter != labels.end(); iter++) 
   * { cout << *iter << " "; } 
   */ 
  read_Mnist_Images("train-images.idx3-ubyte", images);//读取训练集手写字体图像 
  /*:
  for (int i = 0; i < images.size(); i++) 
  { 
    for (int j = 0; j < images[0].size(); j++) 
    { 
      cout << images[i][j] << " "; 
    } 
  } 
  */
  //cout<< images.size()<<endl<<images[0].size()<<endl;



  //  BP网络   //


  double rate1 = 0.0033,rate2=0.485;//学习率
  double e=0.0;
  double etemp=3.0;
  int times=5000;//训练次数
  cout<<"请输入："<<endl;
  cin>>times;
  double w1[785][ma+1],w2[ma+1][11];//权值数组
  double sig1[ma+1],sig2[11];//记录激活函数输出的数组

  double pl1[ma+1],pl2[11];//记录激活函数的输入的数组
  
  double d[11];//记录label的one-hot编码

  //随机初始化权值
  for(int i=0;i<785;i++)
    for(int j=0;j<ma+1;j++)
      w1[i][j]=0.1*rand()/double(RAND_MAX)-0.05;


  for(int i=0;i<ma+1;i++)
    for(int j=0;j<11;j++)
      w2[i][j]=0.1*rand()/double(RAND_MAX)-0.05;



//
  for(int u=0;u<times;u++)//第u次训练
  {
    //初始化各数组
    for(int j=0;j<ma+1;j++)
    {
      pl1[j]=0.0;
      sig1[j]=0.0;
    }
    for(int i=0;i<11;i++)
    {
      pl2[i]=0.0;
      sig2[i]=0.0;
    }

    for(int tmp=0;tmp<10;tmp++)//one-hot编码label
    {
      if(labels[u]==tmp)
        d[tmp]=1.0;
      else
        d[tmp]=0.0;
    }

    for(int j=0;j<ma;j++)//计算第j个隐含层的加权求和的输出
    {

      for(int k=0;k<784;k++)
      {
        pl1[j]+=w1[k][j]*images[u][k];
      }
      sig1[j]=sigmoid(pl1[j]);

    }

    for(int i=0;i<10;i++)//计算输出层的加权求和
    {
      
      for(int j=0;j<ma;j++)
      {
        
        pl2[i]+=w2[j][i]*sig1[j];
      }
      sig2[i]=sigmoid(pl2[i]);
      



    }


//自适应学习率
  double etempcurrent=0.0;//本次训练的误差平方和
  for(int i=0;i<10;i++)
  {
    double t=sig2[i]-d[i];
    etempcurrent += t*t;
  }
  if(etempcurrent<0.7*etemp)
  {
    rate1*=1.001;
    rate2*=1.002;
  }
  else if(etempcurrent>12*etemp)
  {
    rate1*=0.95;
    rate2*=0.99;
  }
  etemp=etempcurrent;
  e+=etemp;
    //
  if(etemp/10.0<0.0001)//当误差足够小时，不反传误差
    continue;
  if(e/(10.0*u)<0.01)//当平均误差足够小时，停止学习
    break;


//反传误差，调整权值
    double temp[ma+1][11];
    double part1[11];
    for(int i=0;i<10;i++)
    {
      
      part1[i]=sig2[i]*(1.0-sig2[i])*(sig2[i]-d[i]);

      for(int j=0;j<ma;j++)//调整权值
      {
        temp[j][i]=w2[j][i];
        w2[j][i]-=rate2*sig1[j]*part1[i];

      }
    }

    for(int j=0;j<ma;j++)
    {
      double part2=0.0;
      for(int i=0;i<10;i++)
      {
        part2+=temp[j][i]*part1[i];
      }
      double part3=sig1[j]*(1.0-sig1[j]);
      for(int k=0;k<784;k++)
      {
        w1[k][j]-=rate1*images[u][k]*part3*part2;
      }

    }

  }

  //在测试集上进行测试

  read_Mnist_Label("t10k-labels.idx1-ubyte", label);//读取测试集的label
  /* for (auto iter = labels.begin(); iter != labels.end(); iter++) 
   * { cout << *iter << " "; } 
   */ 
  read_Mnist_Images("t10k-images.idx3-ubyte", image);//读取测试集的图像数据 

  int per=0;//记录正确识别个数

  for(int u=0;u<100;u++)//进行100次测试
  {

    for(int j=0;j<ma+1;j++)
    {
      pl1[j]=0.0;
      sig1[j]=0.0;
    }
    for(int i=0;i<11;i++)
    {
      pl2[i]=0.0;
      sig2[i]=0.0;
    }
    for(int j=0;j<ma;j++)//计算第j个隐含层的加权求和的输出
    {

      for(int k=0;k<784;k++)
      {
        pl1[j]+=w1[k][j]*image[u][k];
      }
      sig1[j]=sigmoid(pl1[j]);

    }

    for(int i=0;i<10;i++)//计算输出层的加权求和
    {
      
      for(int j=0;j<ma;j++)
      {
        
        pl2[i]+=w2[j][i]*sig1[j];
      }
      sig2[i]=sigmoid(pl2[i]);
      



    }
    int result=0;//用result记录用bp学习后的预测结果
    double t=-0.1;
    for(int i=0;i<10;i++)
    {
      if(sig2[i]>t)
      {
        result=i;
        t=sig2[i];
      }
    }

    if(result==label[u])//对比结果是否正确
      per++;
    //cout<<result;
    //if((u+1)%10!=0)
    //  cout<<" ";
   // else
    //  cout<<endl;
  }
  
  printf("%.2lf%%\n",(double)per);//输出识别率

  return 0; 
}

