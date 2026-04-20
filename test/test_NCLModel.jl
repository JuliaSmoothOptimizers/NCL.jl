function test_NCLModel()
  # Test parameters
  ρ = 1.0
  y = [2.0, 1.0]
  g = Vector{Float64}(undef, 4)
  cx = Vector{Float64}(undef, 4)

  hrows = [1, 2, 2, 3, 4]
  hcols = [1, 1, 2, 3, 4]
  hvals = Vector{Float64}(undef, 5)
  Hv = Vector{Float64}(undef, 4)

  jrows = [1, 2, 3, 4, 1, 2, 3, 4, 3, 4]
  jcols = [1, 1, 1, 1, 2, 2, 2, 2, 3, 4]
  jvals = Vector{Float64}(undef, 10)
  Jv = Vector{Float64}(undef, 4)

  # Test problem
  f(x) = x[1] + x[2]
  x0 = [0.5, 0.5]
  lvar = [0.0, 0.0]
  uvar = [1.0, 1.0]

  lcon = [-0.5, -1.0, -Inf, 0.5]
  ucon = [Inf, 2.0, 0.5, 0.5]
  A = sparse([
    1.0 -1.0;
    1.0 -1.0
  ])
  c(x) = [
    x[1]^2 + x[2], # nonlinear range constraint
    x[1] * x[2],   # equality constraint
  ]

  name = "Unit test problem"
  nlp::ADNLPModel = ADNLPModel(f, x0, lvar, uvar, A, c, lcon, ucon; name = name)

  ncl_nlin_res = NCLModel(nlp; resid = 1.0, resid_linear = false, y = [1.0, 1.0])
  ncl_nlin_res.y = y
  ncl_nlin_res.ρ = ρ

  ncl_cons_res = NCLModel(nlp; resid = 1.0, resid_linear = true, y = [1.0, 1.0, 1.0, 1.0])
  ncl_cons_res.ρ = ρ

  @testset "NCLModel. No linear residuals" begin
    @testset "NCLModel struct" begin
      @testset "NCLModel struct information about nlp" begin
        @test ncl_nlin_res.nx == 2
        @test ncl_nlin_res.nr == 2  # 2 nonlinear constraint => 2 residuals
        @test ncl_nlin_res.meta.minimize == true
      end

      @testset "NCLModel struct constant parameters" begin
        @test ncl_nlin_res.meta.nvar == 4 # 2 x, 2 r
        @test ncl_nlin_res.meta.lvar == [0.0, 0.0, -Inf, -Inf] # no bounds for residuals
        @test ncl_nlin_res.meta.uvar == [1.0, 1.0, Inf, Inf]
        @test ncl_nlin_res.meta.x0 == [0.5, 0.5, 1.0, 1.0]
        @test ncl_nlin_res.meta.y0 == [0.0, 0.0, 0.0, 0.0]
        @test ncl_nlin_res.y == y
        @test length(ncl_nlin_res.y) == ncl_nlin_res.nr
        @test ncl_nlin_res.meta.nnzj == nlp.meta.nnzj + 2 # 2 residuals, one for each non linear constraint
        @test ncl_nlin_res.meta.nnzh == nlp.meta.nnzh + 2 # add a digonal of ρ
      end
    end

    @testset "NCLModel f" begin
      @test obj(ncl_nlin_res, [0.0, 0.0, 0.0, 0.0]) == 0.0
      @test obj(ncl_nlin_res, [0.5, 0.5, 0.0, -1.0]) == 1.0 - 1.0 + 0.5 * ρ * 1.0
    end

    @testset "NCLModel ∇f" begin
      @testset "NCLModel grad()" begin
        @test grad(ncl_nlin_res, [0.0, 0.0, 0.0, 0.0]) == [1.0, 1.0, 2.0, 1.0]
        @test grad(ncl_nlin_res, [0.5, 0.5, 0.0, -1.0]) == [1.0, 1.0, 2.0, 1.0 - ρ]
      end

      @testset "NCLModel grad!()" begin
        @test grad!(ncl_nlin_res, [0.0, 0.0, 0.0, 0.0], g) == [1.0, 1.0, 2.0, 1.0]
        @test grad!(ncl_nlin_res, [0.5, 0.5, 0.0, -1.0], zeros(4)) == [1.0, 1.0, 2.0, 1.0 - ρ]
      end
    end

    @testset "NCLModel Hessian of the Lagrangian" begin
      @testset "NCLModel Hessian of the Lagrangian hess()" begin
        @test hess(ncl_nlin_res, [0.0, 0.0, 0.0, 0.0], zeros(Float64, 4)).data == [
          0.0 0.0 0.0 0.0
          0.0 0.0 0.0 0.0
          0.0 0.0 ρ 0.0
          0.0 0.0 0.0 ρ
        ]
        @test hess(ncl_nlin_res, ncl_nlin_res.meta.x0, [1.0, 1.0, 1.0, 1.0]).data == [
          2.0 0.0 0.0 0.0  # not symmetric because only the lower triangle is returned by hess
          1.0 0.0 0.0 0.0
          0.0 0.0 ρ 0.0
          0.0 0.0 0.0 ρ
        ]
      end

      @testset "NCLModel Hessian of the Lagrangian hess_structure()" begin
        hrows, hcols = hess_structure(ncl_nlin_res)
        @test hrows[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + 2)] == [3, 4]
        @test hcols[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + 2)] == [3, 4]

        hrows_nlin_res, hcols_nlin_res = hess_structure(ncl_nlin_res)
        hrows_nlp, hcols_nlp = hess_structure(ncl_nlin_res.nlp)
        @test hrows_nlin_res == vcat(hrows_nlp, [3, 4])
        @test hcols_nlin_res == vcat(hcols_nlp, [3, 4])
      end

      @testset "NCLModel Hessian of the Lagrangian hess_coord()" begin
        hvals = hess_coord(ncl_nlin_res, [0.0, 0.0, 0.0, 0.0], zeros(Float64, 4))
        @test hvals[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + 2)] == [ρ, ρ]

        hvals = hess_coord(ncl_nlin_res, ncl_nlin_res.meta.x0, [1.0, 1.0, 1.0, 1.0])
        @test hvals[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + 2)] == [ρ, ρ]
      end

      @testset "NCLModel Hessian of the Lagrangian hprod()" begin
        @test hprod(
          ncl_nlin_res,
          ncl_nlin_res.meta.x0,
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 2.0, 3.0, 4.0],
        ) == [4, 1, 3ρ, 4ρ]
      end

      @testset "NCLModel Hessian of the Lagrangian hprod!()" begin
        @test hprod!(
          ncl_nlin_res,
          ncl_nlin_res.meta.x0,
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 2.0, 3.0, 4.0],
          Hv,
        ) == [4, 1, 3ρ, 4ρ]
      end
    end

    @testset "NCLModel constraint" begin
      @testset "NCLModel constraint cons()" begin
        @test size(cons(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0]), 1) == 4
        @test cons(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0]) == [0.0, 0.0, 2.0, 2.0]
        @test cons(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0]) == [0.5, 0.5, 2.5, 1.5]
      end
      @testset "NCLModel constraint cons!()" begin
        @test size(cons!(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0], cx), 1) == 4
        @test cons!(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0], cx) == [0.0, 0.0, 2.0, 2.0]
        @test cons!(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0], cx) == [0.5, 0.5, 2.5, 1.5]
      end
    end

    @testset "NCLModel constraint jacobian" begin
      @testset "NCLModel constraint jac()" begin
        @test Matrix(jac(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0])) == [
          1 -1 0 0
          1 -1 0 0
          2 1 1 0
          1 1 0 1
        ]

        @test Matrix(jac(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0])) == [
          1 -1 0 0
          1 -1 0 0
          2 1 1 0
          0.5 1 0 1
        ]
      end

      @testset "NCLModel constraint jac_structure()" begin
        jrows, jcols = jac_structure(ncl_nlin_res)
        @test jrows[9:10] == [3, 4]
        @test jcols[9:10] == [3, 4]
        jvals = jac_coord(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0])
        @test jvals[9:10] == [1, 1]
      end

      @testset "NCLModel constraint jac_coord!()" begin
        @test jac_coord!(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0], jvals) ==
              [1.0, 1.0, -1.0, -1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        @test jac_coord!(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0], jvals) ==
              [1.0, 1.0, -1.0, -1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0]
      end

      @testset "NCLModel constraint jprod()" begin
        @test jprod(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]) == [0, 0, 4, 3]
        @test jprod(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]) == [-1, -1, 1, 2]
      end

      @testset "NCLModel constraint jprod!()" begin
        @test jprod!(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], Jv) == [0, 0, 4, 3]
        @test jprod!(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], Jv) == [-1, -1, 1, 2]
      end

      @testset "NCLModel constraint jtprod()" begin
        @test jtprod(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]) == [5, 0, 1, 1]
        @test jtprod(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]) == [1.5, 0, 0, 1]
      end

      @testset "NCLModel constraint jtprod!()" begin
        @test jtprod!(ncl_nlin_res, [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], Jv) == [5, 0, 1, 1]
        @test jtprod!(ncl_nlin_res, [1.0, 0.5, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], Jv) ==
              [1.5, 0, 0, 1]
      end
    end
  end

  @testset "NCLModel. All residuals" begin
    @testset "NCLModel struct" begin
      @testset "NCLModel struct information about nlp" begin
        @test ncl_cons_res.nx == 2
        @test ncl_cons_res.nr == 4 # two non linear constraint, so two residuals
        @test ncl_cons_res.meta.minimize == true
      end

      @testset "NCLModel struct constant parameters" begin
        @test ncl_cons_res.meta.nvar == 6 # 2 x, 4 r
        @test ncl_cons_res.meta.lvar == [0.0, 0.0, -Inf, -Inf, -Inf, -Inf] # no bounds for residuals
        @test ncl_cons_res.meta.uvar == [1.0, 1.0, Inf, Inf, Inf, Inf]
        @test ncl_cons_res.meta.x0 == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
        @test ncl_cons_res.meta.y0 == [0.0, 0.0, 0.0, 0.0]
        @test ncl_cons_res.y == [1.0, 1.0, 1.0, 1.0]
        @test length(ncl_cons_res.y) == ncl_cons_res.nr
        @test ncl_cons_res.meta.nnzj == nlp.meta.nnzj + 4 # 2 residuals, one for each constraint
        @test ncl_cons_res.meta.nnzh == nlp.meta.nnzh + 4 # add a digonal of ρ
      end
    end

    @testset "NCLModel f" begin
      @test obj(ncl_cons_res, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == 0.0
      @test obj(ncl_cons_res, [0.5, 0.5, 0.0, -1.0, 0.0, 1.0]) == 1.0 + 0.0 + 0.5 * ρ * (1.0 + 1.0)
    end

    @testset "NCLModel ∇f" begin
      @testset "NCLModel grad()" begin
        @test grad(ncl_cons_res, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        @test grad(ncl_cons_res, [0.5, 0.5, 0.0, -1.0, 0.0, 1.0]) ==
              [1.0, 1.0, 1.0, 1.0 - ρ, 1.0, 1 + ρ]
      end

      @testset "NCLModel grad!()" begin
        @test grad!(ncl_cons_res, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vcat(g, [1, 2])) ==
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        @test grad!(ncl_cons_res, [0.5, 0.5, 0.0, -1.0, 0.0, 1.0], zeros(6)) ==
              [1.0, 1.0, 1.0, 1.0 - ρ, 1.0, 1 + ρ]
      end
    end

    @testset "NCLModel Hessian of the Lagrangian" begin
      @testset "NCLModel Hessian of the Lagrangian hess()" begin
        @test hess(ncl_cons_res, zeros(6), zeros(4)).data == [
          0.0 0.0 0.0 0.0 0.0 0.0
          0.0 0.0 0.0 0.0 0.0 0.0
          0.0 0.0 ρ 0.0 0.0 0.0
          0.0 0.0 0.0 ρ 0.0 0.0
          0.0 0.0 0.0 0.0 ρ 0.0
          0.0 0.0 0.0 0.0 0.0 ρ
        ]
        @test hess(ncl_cons_res, ncl_cons_res.meta.x0, ones(4)).data == [
          2.0 0.0 0.0 0.0 0.0 0.0 # not symmetric because only the lower triangle is returned by hess
          1.0 0.0 0.0 0.0 0.0 0.0
          0.0 0.0 ρ 0.0 0.0 0.0
          0.0 0.0 0.0 ρ 0.0 0.0
          0.0 0.0 0.0 0.0 ρ 0.0
          0.0 0.0 0.0 0.0 0.0 ρ
        ]
      end

      @testset "NCLModel Hessian of the Lagrangian hess_structure()" begin
        hrows, hcols = hess_structure(ncl_cons_res)
        @test hrows[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + ncl_cons_res.nr)] == [3, 4, 5, 6]
        @test hcols[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + ncl_cons_res.nr)] == [3, 4, 5, 6]

        @test hess_coord(ncl_cons_res, zeros(6), zeros(4))[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + ncl_cons_res.nr)] ==
              [ρ, ρ, ρ, ρ]
        @test hess_coord(ncl_cons_res, ncl_cons_res.meta.x0, ones(4))[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + ncl_cons_res.nr)] ==
              [ρ, ρ, ρ, ρ]
      end

      @testset "NCLModel Hessian of the Lagrangian hess_coord!()" begin
        hvals = hess_coord(ncl_cons_res, zeros(6))
        @test hvals[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + ncl_cons_res.nr)] == [ρ, ρ, ρ, ρ]

        hvals = hess_coord(ncl_cons_res, ncl_cons_res.meta.x0, ones(4))
        @test hvals[(nlp.meta.nnzh + 1):(nlp.meta.nnzh + ncl_cons_res.nr)] == [ρ, ρ, ρ, ρ]
      end

      @testset "NCLModel Hessian of the Lagrangian hprod()" begin
        @test hprod(ncl_cons_res, ncl_cons_res.meta.x0, ones(4), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) ==
              [4, 1, 3 * ρ, 4 * ρ, 5 * ρ, 6 * ρ]
      end

      @testset "NCLModel Hessian of the Lagrangian hprod!()" begin
        @test hprod!(
          ncl_cons_res,
          ncl_cons_res.meta.x0,
          ones(4),
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          vcat(Hv, [0.0, 0.0]),
        ) == [4, 1, 3 * ρ, 4 * ρ, 5 * ρ, 6 * ρ]
      end
    end

    @testset "NCLModel constraint" begin
      @testset "NCLModel constraint cons()" begin
        @test size(cons(ncl_cons_res, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]), 1) == 4
        @test cons(ncl_cons_res, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]) == [0.0, 1.0, 3.0, 2.0]
        @test cons(ncl_cons_res, [1.0, 0.5, 1.0, 2.0, 0.0, -1.0]) == [1.5, 2.5, 1.5, -0.5]
      end
      @testset "NCLModel constraint cons!()" begin
        @test size(cons!(ncl_cons_res, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0], cx), 1) == 4
        @test cons!(ncl_cons_res, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0], cx) == [0.0, 1.0, 3.0, 2.0]
        @test cons!(ncl_cons_res, [1.0, 0.5, 1.0, 2.0, 0.0, -1.0], cx) == [1.5, 2.5, 1.5, -0.5]
      end
    end

    @testset "NCLModel constraint jacobian" begin
      @testset "NCLModel constraint jac()" begin
        @test Matrix(jac(ncl_cons_res, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0])) == [
          1 -1 1 0 0 0
          1 -1 0 1 0 0
          2 1 0 0 1 0
          1 1 0 0 0 1
        ]

        @test Matrix(jac(ncl_cons_res, [1.0, 0.5, 1.0, 1.0, 0.0, -1.0])) == [
          1 -1 1 0 0 0
          1 -1 0 1 0 0
          2 1 0 0 1 0
          0.5 1 0 0 0 1
        ]
      end

      @testset "NCLModel constraint jac_coord()" begin
        # Our home-baked jac_structure doesn't return elements in the same order
        # as the default implementation in NLPModels.
        rows, cols = jac_structure(ncl_cons_res)
        @test rows[9:12] == [1, 2, 3, 4]
        @test cols[9:12] == [3, 4, 5, 6]
        vals = jac_coord(ncl_cons_res, [1.0, 0.5, 1.0, 1.0, 0.0, -1.0])
        @test vals[9:12] == [1, 1, 1, 1]
      end

      @testset "NCLModel constraint jac_coord!()" begin
        rows, cols = jac_structure(ncl_cons_res)
        x = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        vals = Vector{Float64}(undef, ncl_cons_res.meta.nnzj)
        jac_coord!(ncl_cons_res, x, vals)
        @test vals == [1, 1, -1, -1, 2, 1, 1, 1, 1, 1, 1, 1]
        x = [1.0, 0.5, 1.0, 1.0, 0.0, -1.0]
        jac_coord!(ncl_cons_res, x, vals)
        @test vals == [1, 1, -1, -1, 2, 0.5, 1, 1, 1, 1, 1, 1]
      end

      @testset "NCLModel constraint jac_lin_coord()" begin
        rows, cols = jac_lin_structure(ncl_cons_res)
        @test rows[5:6] == [1, 2]
        @test cols[5:6] == [3, 4]
        vals = jac_lin_coord(ncl_cons_res, [1.0, 0.5, 1.0, 1.0, 0.0, -1.0])
        @test vals[5:6] == [1, 1]
      end
      @testset "NCLModel constraint jac_nln_coord()" begin
        rows, cols = jac_nln_structure(ncl_cons_res)
        @test rows[5:6] == [1, 2]
        @test cols[5:6] == [5, 6]
        vals = jac_nln_coord(ncl_cons_res, [1.0, 0.5, 1.0, 1.0, 0.0, -1.0])
        @test vals[5:6] == [1, 1]
      end

      @testset "NCLModel constraint jprod()" begin
        @test jprod(ncl_cons_res, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) ==
              [1, 1, 4, 3]
        @test jprod(
          ncl_cons_res,
          [1.0, 0.5, 1.0, 1.0, 0.0, -1.0],
          [0.0, 1.0, 0.0, 1.0, -1.0, -1.0],
        ) == [-1, 0, 0, 0]
      end

      @testset "NCLModel constraint jprod!()" begin
        @test jprod!(
          ncl_cons_res,
          [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          Jv,
        ) == [1, 1, 4, 3]
        @test jprod!(
          ncl_cons_res,
          [1.0, 0.5, 1.0, 1.0, 0.0, -1.0],
          [0.0, 1.0, 0.0, 1.0, -1.0, -1.0],
          Jv,
        ) == [-1, 0, 0, 0]
      end

      @testset "NCLModel constraint jtprod()" begin
        @test jtprod(ncl_cons_res, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]) ==
              [5, 0, 1, 1, 1, 1]
        @test jtprod(ncl_cons_res, [1.0, 0.5, 1.0, 1.0, 0.0, -1.0], [0.0, 1.0, 0.0, 1.0]) ==
              [1.5, 0, 0, 1, 0, 1]
      end

      @testset "NCLModel constraint jtprod!()" begin
        @test jtprod!(
          ncl_cons_res,
          [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          vcat(Jv, [0, 1]),
        ) == [5, 0, 1, 1, 1, 1]
        @test jtprod!(
          ncl_cons_res,
          [1.0, 0.5, 1.0, 1.0, 0.0, -1.0],
          [0.0, 1.0, 0.0, 1.0],
          vcat(Jv, [0, 1]),
        ) == [1.5, 0, 0, 1, 0, 1]
      end
    end
  end

  @testset "NLPModelsTest gradient check, resid_linear = false" begin
    g_errs = gradient_check(ncl_nlin_res, x = [1.0, -1.0, 1.0, -1.0])
    @test length(g_errs) == 0
    g_errs = gradient_check(ncl_nlin_res, x = [1.0, 0.5, 0.0, 1.0])
    @test length(g_errs) == 0
  end

  @testset "NLPModelsTest Jacobian check, resid_linear = false" begin
    j_errs = jacobian_check(ncl_nlin_res, x = [1.0, -1.0, 1.0, -1.0])
    @test length(j_errs) == 0
    j_errs = jacobian_check(ncl_nlin_res, x = [1.0, 0.5, 0.0, 1.0])
    @test length(j_errs) == 0
  end

  @testset "NLPModelsTest Hessian check, resid_linear = false" begin
    h_errs = hessian_check_from_grad(ncl_nlin_res, x = [1.0, -1.0, 1.0, -1.0])
    for k ∈ keys(h_errs)
      @test length(h_errs[k]) == 0
    end
    h_errs = hessian_check_from_grad(ncl_nlin_res, x = [1.0, 0.5, 0.0, 1.0])
    for k ∈ keys(h_errs)
      @test length(h_errs[k]) == 0
    end
  end

  @testset "NLPModelsTest gradient check, resid_linear = true" begin
    g_errs = gradient_check(ncl_cons_res, x = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    @test length(g_errs) == 0
    g_errs = gradient_check(ncl_cons_res, x = [1.0, 0.5, 1.0, 1.0, 0.0, -1.0])
    @test length(g_errs) == 0
  end

  @testset "NLPModelsTest Jacobian check, resid_linear = true" begin
    j_errs = jacobian_check(ncl_cons_res, x = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    @test length(j_errs) == 0
    j_errs = jacobian_check(ncl_cons_res, x = [1.0, 0.5, 1.0, 1.0, 0.0, -1.0])
    @test length(j_errs) == 0
  end

  @testset "NLPModelsTest Hessian check, resid_linear = true" begin
    h_errs = hessian_check_from_grad(ncl_cons_res, x = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    for k ∈ keys(h_errs)
      @test length(h_errs[k]) == 0
    end
    h_errs = hessian_check_from_grad(ncl_cons_res, x = [1.0, 0.5, 1.0, 1.0, 0.0, -1.0])
    for k ∈ keys(h_errs)
      @test length(h_errs[k]) == 0
    end
  end

  @testset "NLPModelsTest dimension check, resid_linear = false, linear_api = true" begin
    check_nlp_dimensions(ncl_nlin_res, linear_api = true)
  end

  @testset "NLPModelsTest dimension check, resid_linear = true, linear_api = true" begin
    check_nlp_dimensions(ncl_cons_res, linear_api = true)
  end

  @testset "NLPModelsTest dimension check, resid_linear = false, linear_api = false" begin
    check_nlp_dimensions(ncl_nlin_res, linear_api = false)
  end

  @testset "NLPModelsTest dimension check, resid_linear = true, linear_api = false" begin
    check_nlp_dimensions(ncl_cons_res, linear_api = false)
  end
end
