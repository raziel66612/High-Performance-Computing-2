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


//Print all Vector in loop
// for (i=0; i<n ;i++ ){
// PetscPrintf(comm, "%.2g  ",a[i]);
// PetscPrintf(comm, "\n");
// }


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