function test_NLCModel()
    # Test parameters
    ρ = 1.
    y = [2., 1.]
    g = Vector{Float64}(undef,4)
    cx = Vector{Float64}(undef,4)

    hrows = [1, 2, 2, 3, 4]
    hcols = [1, 1, 2, 3, 4]
    hvals = Vector{Float64}(undef,5)
    Hv = Vector{Float64}(undef,4)

    jrows = [1, 2, 3, 4, 1, 2, 3, 4, 3, 4]
    jcols = [1, 1, 1, 1, 2, 2, 2, 2, 3, 4]
    jvals = Vector{Float64}(undef,10)
    Jv = Vector{Float64}(undef,4)

    # Test problem
    f(x) = x[1] + x[2]
    x0 = [0.5, 0.5]
    lvar = [0., 0.]
    uvar = [1., 1.]

    lcon = [-0.5, -1., -Inf, 0.5]
    ucon = [ Inf,  2.,  0.5, 0.5]
    c(x) = [x[1]   - x[2],   # linear
            x[1]^2 + x[2], # nonlinear range constraint
            x[1]   - x[2],   # linear, lower bounded
            x[1]   * x[2]]   # equality constraint

    name = "Unit test problem"
    nlp::ADNLPModel = ADNLPModel(f, x0 ; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name, lin = [1,3])
    nlc_nlin_res = NCLModel(nlp ; resid = 1., resid_linear = false, y = [1.,1.])

    nlc_nlin_res.y = y
    nlc_nlin_res.ρ = ρ

    nlc_cons_res = NCLModel(nlp ; resid = 1., resid_linear = true, y = [1.,1.,1.,1.])
    nlc_cons_res.ρ = ρ

    @testset "NCLModel. No linear residuals" begin
        @testset "NCLModel struct" begin
            @testset "NCLModel struct information about nlp" begin
                @test nlc_nlin_res.nx == 2
                @test nlc_nlin_res.nr == 2  # 2 nonlinear constraint => 2 residuals
                @test nlc_nlin_res.meta.minimize == true
            end

            @testset "NCLModel struct constant parameters" begin
                @test nlc_nlin_res.meta.nvar == 4 # 2 x, 2 r
                @test nlc_nlin_res.meta.lvar == [0., 0., -Inf, -Inf] # no bounds for residuals
                @test nlc_nlin_res.meta.uvar == [1., 1., Inf, Inf]
                @test nlc_nlin_res.meta.x0 == [0.5, 0.5, 1., 1.]
                @test nlc_nlin_res.meta.y0 == [0., 0., 0., 0.]
                @test nlc_nlin_res.y == y
                @test length(nlc_nlin_res.y) == nlc_nlin_res.nr
                @test nlc_nlin_res.meta.nnzj == nlp.meta.nnzj + 2 # 2 residuals, one for each non linear constraint
                @test nlc_nlin_res.meta.nnzh == nlp.meta.nnzh + 2 # add a digonal of ρ
            end
        end

        @testset "NCLModel f" begin
            @test obj(nlc_nlin_res, [0., 0., 0., 0.]) == 0.
            @test obj(nlc_nlin_res, [0.5, 0.5, 0., -1.]) == 1. - 1. + 0.5 * ρ * 1.
        end

        @testset "NCLModel ∇f" begin
            @testset "NCLModel grad()" begin
                @test grad(nlc_nlin_res, [0., 0., 0., 0.]) == [1., 1., 2., 1.]
                @test grad(nlc_nlin_res, [0.5, 0.5, 0., -1.]) == [1., 1., 2., 1. - ρ]
            end

            @testset "NCLModel grad!()" begin
                @test grad!(nlc_nlin_res, [0., 0., 0., 0.], g) == [1., 1., 2., 1.]
                @test grad!(nlc_nlin_res, [0.5, 0.5, 0., -1.], zeros(4)) == [1., 1., 2., 1. - ρ]
            end
        end

        @testset "NCLModel Hessian of the Lagrangian" begin
            @testset "NCLModel Hessian of the Lagrangian hess()" begin
                @test hess(nlc_nlin_res, [0., 0., 0., 0.], zeros(Float64,4)) == [0. 0. 0. 0. ;
                                                                                 0. 0. 0. 0. ;
                                                                                 0. 0. ρ  0. ;
                                                                                 0. 0. 0. ρ]
                @test hess(nlc_nlin_res, nlc_nlin_res.meta.x0, [1.,1.,1.,1.]) == [2. 0. 0. 0. ;  # not symmetric because only the lower triangle is returned by hess
                                                                                  1. 0. 0. 0. ;
                                                                                  0. 0. ρ  0. ;
                                                                                  0. 0. 0. ρ]
            end

            @testset "NCLModel Hessian of the Lagrangian hess_structure()" begin
                hrows, hcols = hess_structure(nlc_nlin_res)
                @test hrows[nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]
                @test hcols[nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]

                hrows_nlin_res, hcols_nlin_res = hess_structure(nlc_nlin_res)
                hrows_nlp, hcols_nlp = hess_structure(nlc_nlin_res.nlp)
                @test hrows_nlin_res == vcat(hrows_nlp, [3, 4])
                @test hcols_nlin_res == vcat(hcol_nlp, [3, 4])
            end

            @testset "NCLModel Hessian of the Lagrangian hess_coord()" begin
                hvals = hess_coord(nlc_nlin_res, [0., 0., 0., 0.], zeros(Float64, 4))
                @test hvals[nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]

                hvals = hess_coord(nlc_nlin_res, nlc_nlin_res.meta.x0, [1.,1.,1.,1.])
                @test hvals[nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
            end

            @testset "NCLModel Hessian of the Lagrangian hprod()" begin
                @test hprod(nlc_nlin_res, nlc_nlin_res.meta.x0, [1.,1.,1.,1.], [1.,2.,3.,4.]) == [4, 1, 3ρ, 4ρ]
            end

            @testset "NCLModel Hessian of the Lagrangian hprod!()" begin
                @test hprod!(nlc_nlin_res, nlc_nlin_res.meta.x0, [1.,1.,1.,1.], [1.,2.,3.,4.], Hv) == [4, 1, 3ρ, 4ρ]
            end
        end

        @testset "NCLModel constraint" begin
            @testset "NCLModel constraint cons()" begin
                @test size(cons(nlc_nlin_res, [1.,1.,0.,1.]), 1) == 4
                @test cons(nlc_nlin_res, [1.,1.,0.,1.]) == [0.,2.,0.,2.]
                @test cons(nlc_nlin_res, [1.,0.5,1.,1.]) == [0.5,2.5,0.5,1.5]
            end
            @testset "NCLModel constraint cons!()" begin
                @test size(cons!(nlc_nlin_res, [1.,1.,0.,1.], cx), 1) == 4
                @test cons!(nlc_nlin_res, [1.,1.,0.,1.], cx) == [0.,2.,0.,2.]
                @test cons!(nlc_nlin_res, [1.,0.5,1.,1.], cx) == [0.5,2.5,0.5,1.5]
            end
        end

        @testset "NCLModel constraint jacobian" begin
            @testset "NCLModel constraint jac()" begin
                @test jac(nlc_nlin_res, [1.,1.,0.,1.]) == [1 -1 0 0 ;
                                                2  1 1 0 ;
                                                1 -1 0 0 ;
                                                1  1 0 1 ]

                @test jac(nlc_nlin_res, [1.,0.5,1.,1.]) == [ 1 -1  0  0 ;
                                                    2  1  1  0 ;
                                                    1 -1  0  0 ;
                                                    0.5 1  0  1]
            end

            @testset "NCLModel constraint jac_structure()" begin
                jrows, jcols = jac_coord(nlc_nlin_res, [1., 1., 0., 1.])
                @test jrows[9:10] == [2,4]
                @test jcols[9:10] == [3,4]
                # @test jac_coord(nlc_nlin_res, [1.,0.5,1.,1.])[3][9:10] == [1,1]
            end

            @testset "NCLModel constraint jac_coord!()" begin
                @test jac_coord!(nlc_nlin_res, [1., 1., 0., 1.], jvals) == [1, 2, 1, 1, -1, 1, -1, 1, 1, 1]
                @test jac_coord!(nlc_nlin_res, [1., 0.5, 1., 1.], jvals) == [1, 2, 1, 0.5, -1, 1, -1, 1, 1, 1]
            end

            @testset "NCLModel constraint jprod()" begin
                @test jprod(nlc_nlin_res, [1., 1., 0., 1.], [1., 1., 1., 1.]) == [0, 4, 0, 3]
                @test jprod(nlc_nlin_res, [1., 0.5, 1., 1.], [0., 1., 0., 1.]) == [-1, 1, -1, 2]
            end

            @testset "NCLModel constraint jprod!()" begin
                @test jprod!(nlc_nlin_res, [1., 1., 0., 1.], [1., 1., 1., 1.], Jv) == [0, 4, 0, 3]
                @test jprod!(nlc_nlin_res, [1., 0.5, 1., 1.], [0., 1., 0., 1.], Jv) == [-1, 1, -1, 2]
            end

            @testset "NCLModel constraint jtprod()" begin
                @test jtprod(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.]) == [5,0,1,1]
                @test jtprod(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.]) == [2.5,2,1,1]
            end

            @testset "NCLModel constraint jtprod!()" begin
                @test jtprod!(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.], Jv) == [5,0,1,1]
                @test jtprod!(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.], Jv) == [2.5,2,1,1]
            end
        end
    end

    @testset "NCLModel. All residuals" begin
        @testset "NCLModel struct" begin
            @testset "NCLModel struct information about nlp" begin
                @test nlc_cons_res.nx == 2
                @test nlc_cons_res.nr == 4 # two non linear constraint, so two residuals
                @test nlc_cons_res.meta.minimize == true
            end

            @testset "NCLModel struct constant parameters" begin
                @test nlc_cons_res.meta.nvar == 6 # 2 x, 4 r
                @test nlc_cons_res.meta.lvar == [0., 0., -Inf, -Inf, -Inf, -Inf] # no bounds for residuals
                @test nlc_cons_res.meta.uvar == [1., 1., Inf, Inf, Inf, Inf]
                @test nlc_cons_res.meta.x0 == [0.5, 0.5, 1., 1., 1., 1.]
                @test nlc_cons_res.meta.y0 == [0., 0., 0., 0.]
                @test nlc_cons_res.y == [1., 1., 1., 1.]
                @test length(nlc_cons_res.y) == nlc_cons_res.nr
                @test nlc_cons_res.meta.nnzj == nlp.meta.nnzj + 4 # 2 residuals, one for each constraint
                @test nlc_cons_res.meta.nnzh == nlp.meta.nnzh + 4 # add a digonal of ρ
            end
        end

        @testset "NCLModel f" begin
            @test obj(nlc_cons_res, [0., 0., 0., 0., 0., 0.]) == 0.
            @test obj(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.]) == 1. + 0. + 0.5 * ρ * (1. + 1.)
        end

        @testset "NCLModel ∇f" begin
            @testset "NCLModel grad()" begin
                @test grad(nlc_cons_res, [0., 0., 0., 0., 0., 0.]) == [1., 1., 1., 1., 1., 1.]
                @test grad(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.]) == [1., 1., 1., 1. - ρ, 1., 1 + ρ]
            end

            @testset "NCLModel grad!()" begin
                @test grad!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(g, [1,2])) == [1., 1., 1., 1., 1., 1.]
                @test grad!(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.], zeros(6)) == [1., 1., 1., 1. - ρ, 1., 1 + ρ]
            end
        end

        @testset "NCLModel Hessian of the Lagrangian" begin
            @testset "NCLModel Hessian of the Lagrangian hess()" begin
                @test hess(nlc_cons_res, [0., 0., 0., 0.], y=zeros(Float64,6)) == [0. 0. 0. 0. 0. 0. ;
                                                                                   0. 0. 0. 0. 0. 0. ;
                                                                                   0. 0. ρ  0. 0. 0. ;
                                                                                   0. 0. 0. ρ  0. 0. ;
                                                                                   0. 0. 0. 0. ρ  0. ;
                                                                                   0. 0. 0. 0. 0. ρ ]
                @test hess(nlc_cons_res, nlc_cons_res.meta.x0, y=[1.,1.,1.,1.]) == [2. 0. 0. 0. 0. 0. ; # not symmetric because only the lower triangle is returned by hess
                                                                                    1. 0. 0. 0. 0. 0. ;
                                                                                    0. 0. ρ  0. 0. 0. ;
                                                                                    0. 0. 0. ρ  0. 0. ;
                                                                                    0. 0. 0. 0. ρ  0. ;
                                                                                    0. 0. 0. 0. 0. ρ ]
            end

            @testset "NCLModel Hessian of the Lagrangian hess_structure()" begin
                hrows, hcols = hess_structure(nlc_cons_res)
                @test hrows[nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nr] == [3, 4, 5, 6]
                @test hcols[nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nr] == [3, 4, 5, 6]

                @test hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.], y = zeros(Float64,6))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nr] == [ρ, ρ, ρ, ρ]
                @test hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, y = [1.,1.,1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nr] == [ρ, ρ, ρ, ρ]
            end

            @testset "NCLModel Hessian of the Lagrangian hess_coord!()" begin
                hvals = hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.])
                @test hvals[nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nr] == [ρ, ρ, ρ, ρ]

                hvals = hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, [1.,1.,1.,1.,1.,1.])
                @test hvals[nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nr] == [ρ, ρ, ρ, ρ]
            end

            @testset "NCLModel Hessian of the Lagrangian hprod()" begin
                @test hprod(nlc_cons_res, nlc_cons_res.meta.x0, [1.,2.,3.,4.,5.,6.], y = [1.,1.,1.,1.,1.,1.]) == [4,1,3*ρ,4*ρ,5*ρ,6*ρ]
            end

            @testset "NCLModel Hessian of the Lagrangian hprod!()" begin
                @test hprod!(nlc_cons_res, nlc_cons_res.meta.x0, [1.,2.,3.,4.,5.,6.], y = [1.,1.,1.,1.,1.,1.], vcat(Hv, [0.,0.])) == [4,1,3*ρ,4*ρ,5*ρ,6*ρ]
            end
        end

        @testset "NCLModel constraint" begin
            @testset "NCLModel constraint cons()" begin
                @test size(cons(nlc_cons_res, [1.,1.,0.,1.,1.,1.]), 1) == 4
                @test cons(nlc_cons_res, [1.,1.,0.,1.,1.,1.]) == [0.,3.,1.,2.]
                @test cons(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.]) == [1.5,2.5,0.5,-0.5]
            end
            @testset "NCLModel constraint cons!()" begin
                @test size(cons!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], cx), 1) == 4
                @test cons!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], cx) == [0.,3.,1.,2.]
                @test cons!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], cx) == [1.5,2.5,0.5,-0.5]
            end
        end

        @testset "NCLModel constraint jacobian" begin
            @testset "NCLModel constraint jac()" begin
                @test jac(nlc_cons_res, [1.,1.,0.,1.,1.,1.]) == [1 -1  1  0  0  0;
                                                                2  1  0  1  0  0;
                                                                1 -1  0  0  1  0;
                                                                1  1  0  0  0  1]

                @test jac(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.]) == [1  -1  1  0  0  0;
                                                                2   1  0  1  0  0;
                                                                1  -1  0  0  1  0;
                                                                0.5  1  0  0  0  1]
            end

            @testset "NCLModel constraint jac_coord()" begin
                @test jac_coord(nlc_cons_res, [1.,1.,0.,1.,1.,1.])[1][9:12] == [1,2,3,4]
                @test jac_coord(nlc_cons_res, [1.,1.,0.,1.,1.,1.])[2][9:12] == [3,4,5,6]
                @test jac_coord(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.])[3][9:12] == [1,1,1,1]
            end

            @testset "NCLModel constraint jac_coord!()" begin
                rows, cols = jac_structure(nlc_cons_res)
                x = [1.,1.,0.,1.,1.,1.]
                vals = Vector{Float64}(undef, nlc_cons_res.meta.nnzj)
                jac_coord!(nlc_cons_res, x, rows, cols, vals)
                # @show rows vcat(jrows, [1,2])
                # @show cols vcat(jcols, [0,0])
                # @test all(rows .== vcat(jrows, [1,2]))  # FIXME: le membre de droite devrait être [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
                # @test all(cols .== vcat(jcols, [0,0]))  # FIXME: ce test échoue ; comment un indice de colonne pourrait-il être nul ??? Valeur correcte : [1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6]
                @test all(vals .== [1,2,1,1,-1,1,-1,1,1,1,1,1])
                x = [1.,0.5,1.,1.,0.,-1.]
                jac_coord!(nlc_cons_res, x, rows, cols, vals)
                @test all(vals .== [1,2,1,0.5,-1,1,-1,1,1,1,1,1])
            end

            @testset "NCLModel constraint jac_struct()" begin
                @test jac_structure(nlc_cons_res)[1][9:12] == [1,2,3,4]
                @test jac_structure(nlc_cons_res)[2][9:12] == [3,4,5,6]
            end

            @testset "NCLModel constraint jprod()" begin
                @test jprod(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.,1.,1.]) == [1,4,1,3]
                @test jprod(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.,-1.,-1.]) == [-1,2,-2,0]
            end

            @testset "NCLModel constraint jprod!()" begin
                @test jprod!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.,1.,1.], Jv) == [1,4,1,3]
                @test jprod!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.,-1.,-1.], Jv) == [-1,2,-2,0]
            end

            @testset "NCLModel constraint jtprod()" begin
                @test jtprod(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.]) == [5,0,1,1,1,1]
                @test jtprod(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.]) == [2.5,2,0,1,0,1]
            end

            @testset "NCLModel constraint jtprod!()" begin
                @test jtprod!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.], vcat(Jv, [0,1])) == [5,0,1,1,1,1]
                @test jtprod!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.], vcat(Jv, [0,1])) == [2.5,2,0,1,0,1]
            end
        end
    end
end
