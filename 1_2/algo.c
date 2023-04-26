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

PetscInt matrowi,matrown,matcoli,matcoln;

PetscInitialize(&argc, &args, NULL, help);
MPI_Comm comm = PETSC_COMM_WORLD;  

PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-m",&m,&flg));
PetscCall( PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n",&n,&flg));

//creating random values for vector
PetscCall( PetscRandomCreate(comm,&rann));
PetscCall( PetscRandomSetFromOptions(rann));

//Create Vector in Petsc
PetscCall(VecCreate(comm,&a));
PetscCall(VecSetSizes(a,n,PETSC_DECIDE));
PetscCall(VecSetFromOptions(a));
//Duplicate Vectors
PetscCall(VecDuplicate(a,&b));
PetscCall(VecDuplicate(a,&c));
PetscCall(VecDuplicate(a,&d));

//Matrix 
PetscCall(MatCreate(comm,&A));
PetscCall(MatSetSizes(A,m,n,PETSC_DETERMINE,PETSC_DETERMINE));
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


PetscCall(MatGetOwnershipRange(Q,&matrowi,&matrown)); //didnt used these range in this code, but might be useful later.
PetscCall(MatGetOwnershipRangeColumn(Q,&matcoli,&matcoln));

//orthogonalizing: Part 1
for (i=matcoli; i<matcoln; i++)
{
        PetscCall(MatDenseGetColumnVecWrite(Q,i,&a));
        PetscCall(VecAssemblyBegin(a));
        PetscCall(VecAssemblyEnd(a));
        // PetscPrintf(comm,"Vector a in range loop no.,i = %d  is ",i); //-------------------------Print---------------------
        // PetscCall(VecView(a,PETSC_VIEWER_STDOUT_WORLD)); //-------------------------Print---------------------

    for(j=matcoli; j<i; j++)
    {
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
    //Restoring the extracted array (which we converted to basis) back to it original location.
    PetscCall(MatDenseRestoreColumnVecWrite(Q, i, &a));

}
    MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY);  
    MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY); 

//orthogonalizing: Part 2
for (i=n-1; i>=0; i--)
{
        PetscCall(MatDenseGetColumnVecWrite(Q,i,&a));
    for(j=n-1; j>i; j--){
        PetscCall(MatGetColumnVector(A,b,j));
        PetscCall(MatGetColumnVector(Q,c,j));
        PetscCall(VecNorm(b, NORM_2, &normV));
        PetscCall(VecDot(a,b,&dot_prod));
        PetscCall(VecAXPY(a,-dot_prod,c));  //Computes [y = alpha x + y] ..... [VecAXPY(Vec y, PetscScalar alpha,Vec x)]  
    }
    PetscCall(MatDenseRestoreColumnVecWrite(Q, i, &a));
}


//Orthogonality Check
PetscCall( PetscPrintf(PETSC_COMM_WORLD, "\n[A*Q = I]: \n") );   //Print ---------------------------
for (i=0; i<n; i++) {
PetscCall(MatGetColumnVector(A,c,i));
    for (j=0; j<n; j++) {
        PetscScalar dot;
        PetscCall(MatGetColumnVector(Q,d,j));
        // PetscCall(VecAssemblyEnd(d));

        PetscCall( VecDot(c,d,&dot) );
        if(PetscAbsScalar(dot)< 0.000001) dot=0.0;
        PetscCall( PetscPrintf(comm,"%.2g  ",dot) );
        }
    PetscCall( PetscPrintf(comm, "\n") );
}

// Mat B;
// PetscCall(MatMatMult(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B));
// // PetscCall(MatProductCreate(Q,A,NULL,&B));
// MatView(B,PETSC_VIEWER_STDOUT_WORLD); //print matrix Q

//Destroy Matrices
PetscCall(MatDestroy(&A));
PetscCall(MatDestroy(&Q));

//Destroy Vectors
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