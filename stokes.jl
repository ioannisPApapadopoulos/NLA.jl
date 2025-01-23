using Gridap
using MatrixFactorizations, LinearAlgebra
using IterativeSolvers, LinearMaps


"""
This script solves a Stokes lid-driven cavity problem by assembling the Stokes system
and then solving the corresponding linear system with
(i) LU factorization
(ii) Schur complement factorization, where the negative Schur complement is inverted with CG
(iii) Schur complement factorization, where the negative Schur complement is inverted with CG preconditioned with the pressure mass matrix.

"""

function assemble_stokes_system(n::Integer)
    # Helper function for assembling saddle point Stokes problem
    # with a P2-P1 Taylor-Hood discretisation.
    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain, partition)
    model = simplexify(model)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri1",[6,])
    add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

    order = 2
    reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

    V = TestFESpace(model,reffeᵤ,labels=labels,dirichlet_tags=["diri0","diri1"],conformity=:H1)
    Q = TestFESpace(model,reffeₚ,conformity=:H1,constraint=:zeromean)
    Y = MultiFieldFESpace([V,Q])

    u0 = VectorValue(0,0)
    u1 = VectorValue(1,0)
    U = TrialFESpace(V,[u0,u1])
    P = TrialFESpace(Q)
    X = MultiFieldFESpace([U,P])

    degree = order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ,degree)

    f = VectorValue(0.0,0.0)

    a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p - q*(∇⋅u) )dΩ
    l((v,q)) = ∫( v⋅f )dΩ
    op = AffineFEOperator(a,l,X,Y)

    am(p,q) = ∫(p*q)dΩ
    lm(q) = ∫(0.0*q)dΩ
    opm = AffineFEOperator(am,lm,P,Q)


    u0 = interpolate_everywhere(x->VectorValue(0.0,0.0), U).free_values
    p0 = interpolate_everywhere(x->0.0, P).free_values
    zh = FEFunction(X,[u0; p0]);

    b, J = Gridap.Algebra.residual_and_jacobian(op, zh)
    Mp = Gridap.Algebra.jacobian(opm,interpolate_everywhere(x->0.0, P))
    n, m = length(u0), length(p0)
    return -b, J, Mp, (n,m), (U,P,Ωₕ)
end


timings = zeros(5, 3)
results = zeros(Int64, 5, 4)
for (n, i) in zip([20, 40, 80, 160, 320], 1:5)
    print("Considering n=$n.\n")
    b, J, Mp, (n,m), (U,P,Ωₕ) = assemble_stokes_system(n)

    
    results[i,1] = n+m
    fv, gv = b[1:n], b[n+1:end]

    # Extract out submatrices
    A = J[1:n, 1:n]
    B = J[1:n, n+1:end]

    # Cholesky factorization of sparse SPD matrix A
    t0 = @elapsed chol_A = MatrixFactorizations.cholesky(A)

    y = B' * (chol_A \ fv) - gv

    # Solve Stokes system by a direct LU factorization
    t1 = @elapsed lu_J = MatrixFactorizations.lu(J)
    t2 = @elapsed x = lu_J \ b

    # Save solution
    uh, ph = FEFunction(U, x[1:n]), FEFunction(P, x[n+1:end])
    writevtk(Ωₕ,"results",order=2,cellfields=["uh"=>uh,"ph"=>ph])

    # Apply negative Schur complement
    Sf(x) = B' * ( chol_A \ (B*x) )
    Sm = LinearMap(Sf, m; ismutating=false, issymmetric=true, isposdef=true)

    # Solve Schur complement problem with CG
    t3 = @elapsed p1, info = IterativeSolvers.cg(Sm, y, log=true)
    results[i,2] = info.iters

    # Solve Schur complement problem with preconditioned CG
    lu_Mp = MatrixFactorizations.lu(Mp)
    t4 = @elapsed p, info = IterativeSolvers.cg(Sm, y, Pl=lu_Mp, log=true)
    results[i,3] = info.iters


    # Recover velocity
    t5 = @elapsed u = chol_A \ (fv - B*p)
    err = norm([u;p] - x) / norm(x)
    results[i,4] = err

    # Record timings
    timings[i,1] = t1+t2
    timings[i,2] = t0+t3+t5
    timings[i,3] = t0+t4+t5
end
