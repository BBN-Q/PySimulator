/*
 * Some small helper functions
 */

//Helper function to move into the interaction frame defined by a Hamiltonian
inline MatrixXcd move2interaction_frame(const MatrixXcd & Hint, const double & curTime, const MatrixXcd & Hin){
	MatrixXcd transformMat = (i*TWOPI*curTime*Hint).exp();
    return transformMat*Hin*transformMat.adjoint() - Hint;
}

//Helper function to calculate the matrix exponential of a symmetric (Hermitian) matrix multiplied by a constant
//through the eigenvalue decomposition
template <typename MatrixType, typename T>
inline MatrixType expm_eigen(const MatrixType & matIn, const T & mult){
	size_t dim = matIn.rows();
	SelfAdjointEigenSolver<MatrixType> es(matIn);
	MatrixXd D = es.eigenvalues();
	MatrixType V = es.eigenvectors();
//	return V*((mult*D).array().exp().matrix().replicate(1,dim).cwiseProduct(V.adjoint()));
	return V*(mult*D).array().exp().matrix().asDiagonal()*V.adjoint();
}


//Helper function to do the kronecker product.  There is one built into Eigen but you have to pass it answer space and I'd rather take the temporary hit
// and assume we are working with complex matrices
inline MatrixXcd kron(const MatrixXcd & A, const MatrixXcd & B){
	//Some shortcuts to dimensions
	const size_t Ar = A.rows(), Ac = A.cols(), Br = B.rows(), Bc = B.cols();
	//Initialize the output matrix
	MatrixXcd matOut = MatrixXcd::Zero(Ar*Br, Ac*Bc);
	//Now just loop over the usual block multiplication
	for (size_t rowct=0; rowct<Ar; ++rowct)
	   for (size_t colct=0; colct<Ac; ++colct)
	     matOut.block(rowct*Br,colct*Bc,Br,Bc) = A(rowct,colct)*B;

	return matOut;
}


//Helper function to create the column-stack representation of a Lindbladian dissipator
inline MatrixXcd superOp_colStack_dissipator(const MatrixXcd & dissipatorIn){
	size_t dim = dissipatorIn.rows();
	MatrixXcd tmpEye = MatrixXcd::Identity(dim, dim);
	return kron(dissipatorIn.conjugate(), dissipatorIn) -0.5*kron(tmpEye, dissipatorIn.adjoint()*dissipatorIn) -0.5*kron(dissipatorIn.transpose()*dissipatorIn.conjugate(), tmpEye);
}


//Helper function to create the column-stack representation of a Hamiltonian
inline MatrixXcd superOp_colStack_hamiltonian(const MatrixXcd & HamIn){
	size_t dim = HamIn.rows();
	MatrixXcd tmpEye = MatrixXcd::Identity(dim, dim);
	return kron(HamIn.conjugate(), tmpEye) - kron(tmpEye, HamIn);
}
