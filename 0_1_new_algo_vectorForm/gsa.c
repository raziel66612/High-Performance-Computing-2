
#include "petsc.h"
#include <petscmath.h>

static char help[] = "\n\n This code take k vectors of size N and check their orthogonalita by a new algorithm, by creating dual basis, and algorithm is of 2N^{2} order.  \n\n";


int main(int argc,char **args)
{
PetscInt k=4,n=4;
PetscInt i,j; // number of vectors
PetscBool flg;
PetscRandom rann;
PetscScalar dot_prod,normVi,normVj,temp;
Vec a[k],ahat[k];
PetscLogEvent evnt;

PetscInitialize(&argc,&args,NULL,help);

PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-k",&k,&flg) );
PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n",&n,&flg) );

//creating random values for vector
PetscCall( PetscRandomCreate(PETSC_COMM_WORLD,&rann) );
PetscCall( PetscRandomSetFromOptions(rann)  );

PetscCall( VecCreate(PETSC_COMM_WORLD,&a[0]) );
PetscCall( VecSetSizes(a[0],PETSC_DETERMINE,n) );
PetscCall( VecSetFromOptions(a[0])  );

// create k number of vectors
for (i=1; i<k; i++) {
PetscCall( VecDuplicate(a[0], &a[i])  );
}
for (i=0; i<k; i++) {
PetscCall( VecDuplicate(a[0], &ahat[i])  );
}

for (i = 0; i < k; i++) {
PetscCall( VecSetRandom(a[i], rann) );
}
PetscCall( PetscRandomDestroy(&rann) );

// ------------------log ---------------------
PetscCall( PetscLogEventRegister("New_algo_orthogonalization",0,&evnt) ); // log evnt
PetscCall( PetscLogEventBegin(evnt,0,0,0,0) );
//--------------------- orthogonalize the vectors ----------------------
// alternate way 
for (i=0; i<k; i++) {
VecCopy(a[i],ahat[i]);
    for (j=0; j<i; j++) {
        PetscCall( VecNorm(ahat[j],NORM_2,&normVj) );
        PetscCall( VecDot(a[i],ahat[j],&dot_prod) );
            temp=dot_prod/normVj/normVj;
        PetscCall( VecAXPY(ahat[i],-temp,ahat[j]));
    }
    
PetscCall(VecNorm(ahat[i],NORM_2,&normVi) );
PetscCall(VecScale(ahat[i],1.0/(normVi*normVi)));
// PetscPrintf(PETSC_COMM_WORLD," ahat at end of iteration i = %d \n",i); PetscCall(VecView(ahat[i],PETSC_VIEWER_STDOUT_WORLD));
}

for(i=k-1; i>=0 ; i--){
    for(j=k-1; j>i; j--){
        VecDot(ahat[i],a[j],&dot_prod);
        PetscCall( VecNorm(a[j],NORM_2,&normVj) );
        VecAXPY(ahat[i],-dot_prod,ahat[j]);
    }
}

PetscCall( PetscLogEventEnd(evnt,0,0,0,0));

//-------print dot product of vect on terminal to verifa orthogonalita ----------------
for (i=0; i<k; i++) {
    for (j=0; j<k; j++) {
        PetscScalar dot;
        PetscCall( VecDot(a[i],ahat[j],&dot) );

        if(PetscAbsScalar(dot)< 0.00001) dot=0.0;
        PetscCall( PetscPrintf(PETSC_COMM_WORLD,"%.2g  ",dot) );
        }
    PetscCall( PetscPrintf(PETSC_COMM_WORLD, "\n") );
}
   
for (i=0; i<k; i++) {
PetscCall( VecDestroy(&a[i]) );
PetscCall( VecDestroy(&ahat[i]) );
}
PetscCall( PetscFinalize() );
return 0;
}