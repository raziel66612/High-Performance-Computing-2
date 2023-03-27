
#include "petsc.h"
#include <petscmath.h>

static char help[] = "\n\n This code take k vectors of size N and check their orthogonalita ba gram schmidt algorithm \n\n";


int main(int argc,char **args)
{
PetscInt k=4,n=4;
PetscInt i,j; // number of vectors
PetscBool flg;
PetscRandom rann;
PetscScalar dot_prod,normVi,normVj,normVsqi,normVsqj,temp;
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
for (i=1; i<k; ++i) {
PetscCall( VecDuplicate(a[0], &a[i])  );
}
for (i=0; i<k; ++i) {
PetscCall( VecDuplicate(a[0], &ahat[i])  );
}

for (i = 0; i < k; ++i) {
PetscCall( VecSetRandom(a[i], rann) );
// PetscCall(VecView(a[i],PETSC_VIEWER_STDOUT_WORLD));  //----------Print----------------
// PetscCall(VecView(ahat[i],PETSC_VIEWER_STDOUT_WORLD));  //----------Print----------------
}


PetscCall( PetscRandomDestroy(&rann) );

// ------------------log ---------------------
PetscCall( PetscLogEventRegister("Gram_Schmidt_orthogonalization",0,&evnt) ); // log evnt
PetscCall( PetscLogEventBegin(evnt,0,0,0,0) );
//--------------------- orthogonalize the vectors ----------------------
for (i=0; i<k; i++) {
    PetscCall( VecNorm(a[i],NORM_2,&normVi) );   
    normVsqi = PetscSqr(normVi);
    // PetscPrintf(PETSC_COMM_WORLD,"\n In iteration i = %d, ahat before AXPY \n",i);  PetscCall(VecView(ahat[i],PETSC_VIEWER_STDOUT_WORLD));  //----------Print----------------
    PetscCall( VecAXPY(ahat[i],1.0/normVi,a[i]) ); // y= alpha*x+y ,, and here ahat=y, which was initially assigned as 0.
    // PetscPrintf(PETSC_COMM_WORLD,"\n In iteration i = %d, ahat after AXPY \n",i);  PetscCall(VecView(ahat[i],PETSC_VIEWER_STDOUT_WORLD));  //----------Print----------------

    for (j=0; j<k; j++) {
        PetscCall( VecNorm(a[j],NORM_2,&normVj) );
        PetscCall( VecScale(a[j],1.0/normVj) );  // basis of a[j]
  
        PetscCall( VecDot(a[i],a[j],&dot_prod) );
        normVsqj = PetscSqr(normVj);
        temp=dot_prod*normVsqj;
        PetscCall( VecAXPY(ahat[i],-temp,a[j]));
    }

PetscCall(VecScale(ahat[i],1.0/normVsqi));
// PetscPrintf(PETSC_COMM_WORLD," ahat at end of iteration i = %d \n",i); PetscCall(VecView(ahat[i],PETSC_VIEWER_STDOUT_WORLD));
}
for(i=k-2; i>0 ; i--){
    for(j=k-1; j>i+1; j--){
        VecDot(ahat[i],a[j],&dot_prod);
        PetscCall( VecNorm(a[j],NORM_2,&normVj) );
        PetscCall( VecScale(a[j],1.0/normVj) );  // basis of a[j]
        VecAXPY(ahat[i],dot_product,a[j])
    }
}



PetscCall( PetscLogEventEnd(evnt,0,0,0,0) );

//-------print dot product of vect on terminal to verifa orthogonalita ----------------
for (i=0; i<k; i++) {
    for (j=0; j<k; j++) {
        PetscScalar dot;
        PetscCall( VecDot(a[i],a[j],&dot) );
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