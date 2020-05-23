
#include<stdio.h>
#include<stdlib.h>

#include<fftw3.h>
#include<math.h>
#include<complex.h>

float func(double x)      
{
if(x==0)

{

return 1;

}
else
{
return sin(x)/x;
}
}
void main()
{
int xmin,xmax,n,i,j,h,pi;        
double complex z1,z2,z3;                 
FILE *fft;

fft=fopen("akash.txt","w");         

xmin=-30;
xmax=30;
n=1024;
pi=3.15;

float x[n],d,k[n];    

d=(double)(xmax-xmin)/(n-1);


for(i=0;i<n;i++)
{
x[i]=xmin+i*d;
}

for (j = 0; j < n; j++)
  {
    if (j<n/2){
    k[j]=2*pi*j/(n*d);}
    else if (j==n/2){
    k[j]=-pi/d;}
    else
    k[j]=-k[n-j];
  }

fftw_complex w_p[n],tw_q[n];               
fftw_plan p;
for(j=0;j<n;j++)     
{
w_p[j][0]=func(x[j]);
w_p[j][1]=0;
}

p=fftw_plan_dft_1d(n,w_p,tw_q,FFTW_FORWARD,FFTW_ESTIMATE);    

fftw_execute(p);
for(h=0  ;h<n;  h++)
{
z1=cos(2*pi*xmin*k[h])-sin(2*pi*xmin*k[h])*I;
z2=tw_q[h][0]+tw_q[h][1]*I;

z3=d*z1*z2/sqrt(2*pi);          
fprintf(fft,"%f  %e  %e\n",k[h],creal(z3),cimag(z3));     
}

fclose(fft);
fftw_destroy_plan(p);
}	
