#include "petsc.h"
#include <stdio.h>
static char help[] = " in process ";

int main(int argc,char **args)
{

PetscInt m=4, n=4;
PetscRandom rann;
Vec a,b;
Vec c,d; //Vectors to perform orthogonality
PetscBool flg;

PetscInt i=0,j=0;
PetscReal dot_prod,normV,temp;
Mat A,Q;
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
//Duplicate Vectors
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

PetscCall(PetscPrintf(comm,"\nRandom generated Matrix A:\n"));
MatView(A,PETSC_VIEWER_STDOUT_WORLD); //print matrix A

//Duplicate Matrix
PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&Q)); //Duplicate matrix,

PetscCall(PetscPrintf(comm,"\nDuplicating Matrix A; Matrix Q:\n"));
MatView(Q,PETSC_VIEWER_STDOUT_WORLD); //print matrix Q

//orthogonalizing: Part 1
for (i=0; i<n; i++)
{
        PetscCall(MatGetColumnVector(A,a,i));
        PetscCall(VecAssemblyBegin(a));
        PetscCall(VecAssemblyEnd(a));
        // PetscPrintf(comm,"Vector a in range loop no.,i = %d  is ",i); //-------------------------Print---------------------
        // PetscCall(VecView(a,PETSC_VIEWER_STDOUT_WORLD)); //-------------------------Print---------------------

    for(j=0; j<i; j++)
    {
         // Vec b;
        PetscCall(MatGetColumnVector(Q,b,j));
        PetscCall(VecNorm(b, NORM_2, &normV));
        PetscCall(VecDot(a,b,&dot_prod));
        temp=dot_prod/normV/normV;
        PetscCall(VecAXPY(a,-temp,b));  //Computes [y = alpha x + y] ..... [VecAXPY(Vec y, PetscScalar alpha,Vec x)]  
    }
        PetscCall(VecAssemblyBegin(a));
        PetscCall(VecAssemblyEnd(a));

    //normalizing
    PetscCall(VecNorm(a, NORM_2, &normV));
    PetscCall(VecScale(a, 1.0/(normV*normV)));
    // PetscCall(VecSetValue(a, i, normV, INSERT_VALUES));

    PetscCall(VecGetArray(a,&xvals));
    VecGetOwnershipRange(a,&sstart,&eend);
    for (PetscInt k=sstart; k<eend; k++)
    {
    value=xvals[k];
    PetscCall(MatSetValues(Q,1,&k,1,&i,&value,INSERT_VALUES));
    }   
}
    MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY);  
    MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY); 

//orthogonalizing: Part 2
for (i=n-1; i>=0; i--)
{
        PetscCall(MatGetColumnVector(Q,a,i));
        PetscCall(VecAssemblyBegin(a));
        PetscCall(VecAssemblyEnd(a));
    for(j=n-1; j>i; j--){
        PetscCall(MatGetColumnVector(A,b,j));
        PetscCall(MatGetColumnVector(Q,c,j));
        PetscCall(VecNorm(b, NORM_2, &normV));
        PetscCall(VecDot(a,b,&dot_prod));
        PetscCall(VecAXPY(a,-dot_prod,c));  //Computes [y = alpha x + y] ..... [VecAXPY(Vec y, PetscScalar alpha,Vec x)]  
    }
        PetscCall(VecAssemblyBegin(a));
        PetscCall(VecAssemblyEnd(a));

    PetscCall(VecGetArray(a,&xvals));
    VecGetOwnershipRange(a,&sstart,&eend);
    for (PetscInt k=sstart; k<eend; k++)
    {
    value=xvals[k];
    PetscCall(MatSetValues(Q,1,&k,1,&i,&value,INSERT_VALUES));
    }   
}
    MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY);  
    MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY); 

PetscCall(PetscPrintf(comm,"\nAfter Orthogonalizing, Matrix Q:\n"));
MatView(Q,PETSC_VIEWER_STDOUT_WORLD); //print matrix Q

//Orthogonality Check
PetscCall( PetscPrintf(PETSC_COMM_WORLD, "\n[A*Q = I]: \n") );   //Print ---------------------------
for (i=0; i<n; i++) {
PetscCall(MatGetColumnVector(A,d,i));
    for (j=0; j<n; j++) {
        PetscScalar dot;
        PetscCall(MatGetColumnVector(Q,c,j));
        PetscCall(VecAssemblyEnd(d));

        PetscCall( VecDot(c,d,&dot) );
        if(PetscAbsScalar(dot)< 0.000001) dot=0.0;
        PetscCall( PetscPrintf(comm,"%.2g  ",dot) );
        }
    PetscCall( PetscPrintf(comm, "\n") );
}

PetscCall(MatDestroy(&A));
PetscCall(MatDestroy(&Q));
PetscCall(VecDestroy(&a));
PetscCall(VecDestroy(&b));
PetscCall(VecDestroy(&c));
PetscCall(VecDestroy(&d));

//Destroy random
PetscCall(PetscRandomDestroy(&rann));   

PetscFinalize();
printf("\nThats all\n");
 
    return 0;
}