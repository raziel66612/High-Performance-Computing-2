#include "petsc.h"
#include <petscmath.h>


static char help[] = "\n\n This code take k vectors of size N and check their orthogonality by modified gram schmidt algorithm \n\n";

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
  }

  PetscCall( PetscRandomDestroy(&rann) );

  // ------------------log ---------------------
  PetscCall( PetscLogEventRegister("Gram_Schmidt_orthogonalization",0,&evnt) ); // log evnt
  PetscCall( PetscLogEventBegin(evnt,0,0,0,0) );
  //--------------------- orthogonalize the vectors ----------------------
  for (i=0; i<k; i++) {
    PetscCall( VecCopy(a[i], ahat[i]) ); // make a copy of a[i] to ahat[i]
    PetscCall( VecNorm(ahat[i],NORM_2,&normVi) );   
    normVsqi = PetscSqr(normVi);
    PetscCall( VecScale(ahat[i],1.0/normVi) ); // normalize ahat[i]

    for (j=0; j<k; j++) {
      PetscCall( VecDot(ahat[i],a[j],&dot_prod) );
      normVsqj = PetscSqr(normVi);
      temp=dot_prod/normVsqj;
      PetscCall( VecAXPY(ahat[i],-temp,a[j]));
    }

    PetscCall(VecNorm(ahat[i],NORM_2,&normVi));
    PetscCall(VecScale(ahat[i],1.0/normVi));
  }

  PetscCall( PetscLogEventEnd(evnt,0,0,0,0) );

  //-------print dot product of vect on terminal to verify orthogonalization----------------
  for (i=0; i<k; i++) {
    for (j=0; j<k; j++) {
      PetscScalar dot;
      PetscCall( VecDot(ahat[i],ahat[j],&dot) );
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