#include "petsc.h"
#include <stdio.h>
static char help[] = " in process ";

int main(int argc,char **args)
{

PetscInt m=3, n=3;
PetscRandom rann;
Vec a[n];
PetscBool flg;
PetscInt i;
PetscReal dot_prod,normV;
Mat A;

PetscInitialize(&argc, &args, NULL, help);
MPI_Comm comm = PETSC_COMM_WORLD;  


PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-m",&m,&flg));
PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n",&n,&flg));


//creating random values for vector
PetscCall( PetscRandomCreate(comm,&rann));
PetscCall( PetscRandomSetFromOptions(rann));

//Create Vector in Petsc
PetscCall(VecCreate(comm,&a[0]));
PetscCall(VecSetSizes(a[0],PETSC_DETERMINE,n));
PetscCall(VecSetFromOptions(a[0]));

//Create n number of vectors, thus forming a set of vector,matrix
for (i=1; i<m; i++)    //note: i=1, duplicating from a[0]
{
PetscCall(VecDuplicate(a[0], &a[i]));    
}

for (i=0; i<m; i++)
{
    PetscCall(VecSetRandom(a[i], rann));
}

// Print a[0] vector
PetscCall(VecView(a[0], PETSC_VIEWER_STDOUT_WORLD));
PetscPrintf(comm,"\n");

//a_i * b_j = delta_{ij}
for (i=0; i<k; i++) {
    for (j=0; j<i; j++) {
        PetscCall( VecDot(a[i],a[j],&dot_prod) );
        PetscCall( VecAXPY(a[i],-dot_prod,a[j])  );
    }
    PetscCall( VecNorm(a[i],NORM_2,&normV) );
    PetscCall( VecScale(a[i],1.0/normV) );
}
//Print all Vector in loop
// for (i=0; i<n ;i++ ){
// PetscPrintf(comm, "%.2g  ",a[i]);
// PetscPrintf(comm, "\n");
// }

//Create Matrix
// MatCreate(comm,Mat* A) //if A is not defined beofre but you would like to defined here.
MatCreate(comm,&A) // Mat A is predefined before,



// Moore-Penrose comdition W = V^{+}
// V V^{+} V = V
// V^{+} = (V* V)^{-1} V*       where V* is transpose of V




// Destroy duplicate vectors
for (i=0; i<m; i++)  //note: i=0
{  
PetscCall(VecDestroy(&a[i]));
}

//Destroy random
PetscCall(PetscRandomDestroy(&rann));
PetscFinalize();
printf(" Fin ");
 
    return 0;
}