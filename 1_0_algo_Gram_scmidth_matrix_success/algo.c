#include "petsc.h"
#include <stdio.h>
static char help[] = " in process ";

int main(int argc,char **args)
{

PetscInt m=3, n=3;
PetscRandom rann;
Vec a,b;
Vec c,d; //Vectors to perform orthogonality
PetscBool flg;

PetscInt i=0,j=0;
PetscReal dot_prod,normV;
Mat A;
PetscScalar* xvals;
PetscScalar value;
PetscInt sstart,eend;
// PetscInt M,N;

PetscInitialize(&argc, &args, NULL, help);
MPI_Comm comm = PETSC_COMM_WORLD;  


PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-m",&m,&flg));
PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n",&n,&flg));

//creating random values for vector
PetscCall( PetscRandomCreate(comm,&rann));
PetscCall( PetscRandomSetFromOptions(rann));

//Create Vector in Petsc
PetscCall(VecCreate(comm,&a));
PetscCall(VecSetSizes(a,PETSC_DECIDE,n));
PetscCall(VecSetFromOptions(a));

PetscCall(VecDuplicate(a,&b));
PetscCall(VecDuplicate(a,&c));
PetscCall(VecDuplicate(a,&d));
//Matrix 
PetscCall(MatCreate(comm,&A));
PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
PetscCall(MatSetType(A,MATMPIDENSE));
// MatCreateDense(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,NULL,&A);
PetscCall(MatSetFromOptions(A));
PetscCall(MatSetUp(A));
PetscCall(MatSetRandom(A,rann));
MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);  
MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);  
MatView(A,PETSC_VIEWER_STDOUT_WORLD);


//orthogonalizing
for (i=0; i<n; i++)
{
       // Vec a;
        PetscCall(MatGetColumnVector(A,a,i));
        PetscCall(VecAssemblyBegin(a));
        PetscCall(VecAssemblyEnd(a));
        // PetscPrintf(comm,"Vector a in range loop no.,i = %d  is ",i); //-------------------------Print---------------------
        // PetscCall(VecView(a,PETSC_VIEWER_STDOUT_WORLD)); //-------------------------Print---------------------

    for(j=0; j<i; j++)
    {
         // Vec b;
        PetscCall(MatGetColumnVector(A,b,j));
        PetscCall(VecDot(a,b,&dot_prod));
        PetscCall(VecAXPY(a,-dot_prod,b));  //Computes [y = alpha x + y] ..... [VecAXPY(Vec y, PetscScalar alpha,Vec x)]  
    }
        PetscCall(VecAssemblyBegin(a));
        PetscCall(VecAssemblyEnd(a));

    //normalizing
    PetscCall(VecNorm(a, NORM_2, &normV));
    PetscCall(VecScale(a, 1.0/normV));
    // PetscCall(VecSetValue(a, i, normV, INSERT_VALUES));

    PetscCall(VecGetArray(a,&xvals));
    
    VecGetOwnershipRange(a,&sstart,&eend);
    for (PetscInt k=sstart; k<eend; k++)
    {
    PetscCall(value=xvals[k]);
    PetscCall(MatSetValues(A,1,&k,1,&i,&value,INSERT_VALUES));
    }
    
}
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);  
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); 

MatView(A,PETSC_VIEWER_STDOUT_WORLD); //--------------------------Print MAT-------------------

//Orthogonality Check
PetscCall( PetscPrintf(PETSC_COMM_WORLD, "\n \n") );   //Print ---------------------------
for (i=0; i<n; i++) {
    PetscCall(MatGetColumnVector(A,d,i));
    // PetscCall(VecView(c,PETSC_VIEWER_STDOUT_WORLD));//Print --------------


    for (j=0; j<n; j++) {
        PetscScalar dot;
        PetscCall(MatGetColumnVector(A,c,j));
        PetscCall(VecAssemblyEnd(d));

        PetscCall( VecDot(c,d,&dot) );
        if(PetscAbsScalar(dot)< 0.000001) dot=0.0;
        PetscCall( PetscPrintf(comm,"%.2g  ",dot) );
        }
    PetscCall( PetscPrintf(comm, "\n") );
}

PetscCall(MatDestroy(&A));
PetscCall(VecDestroy(&a));
PetscCall(VecDestroy(&b));
PetscCall(VecDestroy(&c));
PetscCall(VecDestroy(&d));

//Destroy random
PetscCall(PetscRandomDestroy(&rann));   

PetscFinalize();
printf("\nThats all folks\n");
 
    return 0;
}