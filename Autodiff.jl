### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ de43006d-f6ad-4e63-af73-d3e7f9a33559
using SymPy                    

# ╔═╡ 29d26680-646b-4ec7-bd17-0d1b874f7c24
using ForwardDiff

# ╔═╡ 58ef52f9-d1c8-45dc-b4af-2c1952baa1ad
using LinearAlgebra

# ╔═╡ 1dd7814a-3f20-420b-8367-463646925a0e
using PlutoUI

# ╔═╡ f911053a-fd18-441d-83bc-82d810c87626
using HypertextLiteral

# ╔═╡ 39da4330-3507-4ae3-857c-e1a929a678d9
using PlutoTest

# ╔═╡ ee7767b5-dea4-47f3-9bd6-042967a01948
begin
	using Plots
	plotly()
end

# ╔═╡ 3abba47d-f2a4-4e8e-a5a4-18aa8724f880
md"""
# Autodiff: *Calculus  from another angle*
(and the special role played by Julia's multiple dispatch and compiler technology)

> *Alan Edelman, 2017*


  The first time I heard about automatic differentiation, it was easy for me to imagine what it was.  I was wrong.  In my head, I thought it was straightforward symbolic differentiation applied to code.  I kind of imagined it was like executing Mathematica or Maple, or even just automatically doing what I learned to do in my calculus class. 

$(Resource("http://www2.bc.cc.ca.us/resperic/math6a/lectures/ch5/1/IntegralTable.gif", :width => 190))

  .... and anyway if it was not that, then it must be finite differences, like one learns in a numerical computing class.
  
$(Resource("http://image.mathcaptain.com/cms/images/122/Diff%202.png", :width => 150))


"""

# ╔═╡ c10103e5-23d3-404a-a3a7-d77ca34f9d83
md"""
## Babylonian `sqrt`

I would like to use a simple example, computation of `sqrt(x)`, where for me how autodiff works came as both a mathematical surprise, and a computing wonder.  The example is  the Babylonian algorithm, known to man for millenia, to compute `sqrt(x)`:  


 > Repeat ``t \leftarrow  (t+x/t) / 2`` until ``t`` converges to ``\sqrt{x}``.
 
Each iteration has one add and two divides. For illustration purposes, 10 iterations suffice.
"""

# ╔═╡ ea58e432-017d-468c-aa0e-8b119f4846d6
md"""
Check that it works:
"""

# ╔═╡ 24af958b-4fe3-40b8-9798-8eabe467eea5
md"""
## ...and now the derivative, almost by magic
"""

# ╔═╡ d870b113-2250-4363-ab6f-4acab6c3fd55
struct D <: Number  # D is a function-derivative pair
    f::Tuple{Float64,Float64}
end

# ╔═╡ ddba9771-006e-4012-9a7a-4839dd7b6d47
let
	d = D((5,6))
	d.f[2]
end

# ╔═╡ f568d9d4-6da8-425e-a0d3-83329ccf72ba
md"""
Sum Rule: `(x+y)' = x' + y'`

Quotient Rule: `(x/y)' = (yx'-xy') / y^2`
"""

# ╔═╡ 07ef3f23-9fdb-406b-88fd-cac39522a18b
begin
	Base.:+(x::D, y::D) = D(x.f .+ y.f)
	Base.:/(x::D, y::D) = D((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
	
	Base.convert(::Type{D}, x::Real) = D((x,zero(x)))
	Base.promote_rule(::Type{D}, ::Type{<:Number}) = D

	# these are used later
	Base.:^(d::D, n::Integer) = D((d.f[1]^n, n*d.f[1]^(n-1) * d.f[2]))
	Base.:^(d::D, n::Real) =    D((d.f[1]^n, n*d.f[1]^(n-1) * d.f[2]))

	Base.:-(x::D, y::D) = D(x.f .- y.f)
	Base.:*(x::D, y::D) = D((x.f[1]*y.f[1], (x.f[2]*y.f[1] + x.f[1]*y.f[2])))
	Base.inv(x::D) = 1/x
end

# ╔═╡ edc5f4de-df01-42d8-9672-ec3c0694bdcd
md"""
## It just works!

How does it work?  We will explain in a moment.  Right now marvel that it does.  Note we did not
import any autodiff package.  Everything is just basic vanilla Julia.
"""

# ╔═╡ dfa4554a-b09d-4d90-83fb-0b0439231e32
md"""
## The assembler

Most folks don't read assembler, but one can see that it is short.
"""

# ╔═╡ 73571d42-53a5-4d02-b2e7-cc4a1e42db2c
md"""
## Symbolically

We haven't yet explained how it works, but it may be of some value to understand that the below is mathematically
equivalent, though not what the computation is doing.

Notice in the below that Babylonian works on SymPy symbols.
"""

# ╔═╡ faff8119-5d7c-4ae5-84df-ca322505adee
x_sym = symbols("x");

# ╔═╡ 89738277-a2ee-4546-902e-4557a1cdc0de
md"""
We can use `x_sym` to do symbolic arithmatic:
"""

# ╔═╡ 460f3ece-5b80-4538-972e-6d7b35ce8195
md"Iterations as a function of ``x``:"

# ╔═╡ 09654e11-9e4e-4690-a60b-71f406afa0a4
md"Derivatives as a function of ``x``:"

# ╔═╡ 6d158dd8-d801-4b17-870e-3103dca2ccb3
md"""
## How autodiff is getting the answer
Let us by hand take the "derivative" of the Babylonian iteration with respect to x. Specifically t′=dt/dx
"""

# ╔═╡ e6177b08-5693-4d67-8eb0-bbf029f143aa
sin(1.0)

# ╔═╡ 010e95d3-a206-4d7b-8dcc-37e7cc46c547
^(2,5)

# ╔═╡ 86f558ed-cbb4-47be-bcd0-1bf10b82a91d
# Base.:^(d::D,n) = (d.f[1]^n, n*d.f[1]^(n-1) * d.f[2])

# ╔═╡ 141df931-1569-43a3-b56e-fd862d181f06
@which sin(1)

# ╔═╡ 25266665-1051-48cd-8c9b-fedb93c1a012
cos(1.0)

# ╔═╡ 7dd8371f-0c1f-43d7-89a7-c4f5244b683b
md"""
What just happened?  Answer: We created an iteration by hand for t′ given our iteration for t. Then we ran the iteration alongside the iteration for t.
"""

# ╔═╡ 1cdfe1b7-af30-4f9a-ac12-63c682be21cd
md"""
How did this work?  It created the same derivative iteration, using very general rules that are set once and need not be written by hand.
"""

# ╔═╡ 0760055e-9d68-48be-b3cd-34b72f42ebad
md"""
Important:: The derivative is substituted before the JIT compiler, and thus efficient compiled code is executed.
"""

# ╔═╡ 46503cf1-95d8-4a9c-98da-c465d66ba0c3
md"""
## Dual Number Notation

Instead of `D(a,b)` we can write ``a + b \epsilon``, where ``\epsilon`` satisfies ``\epsilon^2=0``.  (Some people like to recall imaginary numbers where an ``i`` is introduced with ``i^2=-1``.) 

Others like to think of how engineers just drop the ``\mathcal{O}(\epsilon^2)`` terms.

**The four rules are:**

```math
\begin{align*}
(a+b\epsilon) \pm (c+d\epsilon) &= (a \pm c) +  (b \pm d)\epsilon \\
\\
(a+b\epsilon) * (c+d\epsilon) &= (ac) + (bc+ad)\epsilon \\
\\
(a+b\epsilon) \,/\, (c+d\epsilon) &= (a/c) + (bc-ad)/d^2 \epsilon
\end{align*}
```
"""

# ╔═╡ 91f00d28-d150-468e-88e7-bd67e52d232b
# Base.show(io::IO,x::D) = print(io,x.f[1]," + ",x.f[2]," ϵ")

# ╔═╡ ebf49bda-589f-4389-9a2a-cd5382bd0263
# begin
# 	# Add the last two rules
# 	Base.:-(x::D, y::D) = D(x.f .- y.f)
# 	Base.:*(x::D, y::D) = D((x.f[1]*y.f[1], (x.f[2]*y.f[1] + x.f[1]*y.f[2])))
# end

# ╔═╡ 9baf6cd3-5a63-424a-8576-a1bc5d1b5296
eps_notation(x::D) = Text("$(x.f[1]) + $(x.f[2]) ϵ")

# ╔═╡ c6c0840f-97a5-46ca-9e5c-d711409645ee
D((1,0)) |> eps_notation

# ╔═╡ eef51eb3-f61f-44fb-a5ef-c8092be651bf
D((2,1)) ^2 |> eps_notation

# ╔═╡ 170439e3-ff0b-4100-86f2-5a920c8645da
begin
	ϵ = D((0,1))
end |> eps_notation

# ╔═╡ eaa28814-4ec3-444b-9833-eb57b8a99732
with_terminal() do
	@code_native(ϵ^2)
end

# ╔═╡ 45f42fd1-03e8-4901-8cf7-a10d757c9b96
ϵ * ϵ 

# ╔═╡ 433c0a24-3634-4a1b-9f0b-c9137f5f3dfa
ϵ^2

# ╔═╡ 1c31395e-8c11-4136-af1b-29742834d5d5
md"""
## Generalization to arbitrary roots
"""

# ╔═╡ d5e412cc-8e92-43f5-b28a-87d6b5c14ae0
md"""
## Forward Diff
Now that you understand it, you can use the official package
"""

# ╔═╡ 91c29413-3e2a-4d1a-99d5-1d9a6565ef25
ForwardDiff.derivative(sqrt, 2)

# ╔═╡ d07ac980-809d-4fd7-873e-cb97f8bd1915
@which ForwardDiff.derivative(sqrt, 2)

# ╔═╡ 33ab0a2c-ab42-489c-b3ea-1ef6382ff35d
md"""
## Close Look at Convergence
"""

# ╔═╡ e4f63be3-5fa9-47bb-8d55-2dd9b748cd8f
begin
	struct D1{T} <: Number  # D is a function-derivative pair
	    f::Tuple{T,T}
	end

	Base.:+(x::D1, y::D1) = D1(x.f .+ y.f)
	Base.:/(x::D1, y::D1) = D1((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
	Base.convert(::Type{D1{T}}, x::Real) where {T} = D1((convert(T, x), zero(T)))
	Base.promote_rule(::Type{D1{T}}, ::Type{S}) where {T,S<:Number} = D1{promote_type(T,S)}

end

# ╔═╡ 713ec357-e015-4b8a-9460-75f611dbe500
function Babylonian(x; N = 10) 
    t = (1+x)/2
    for i = 2:N; t=(t + x/t)/2  end    
    t
end  

# ╔═╡ 850675f8-f111-41fc-931e-02dff4851bfd
let
	α = π
	Babylonian(α), √α
end

# ╔═╡ bc255f04-dac3-4cf5-ad67-d323af9642ed
let
	x = 2
	Babylonian(x), √x  # Type \sqrt+<tab> to get the symbol
end

# ╔═╡ 37d2f560-a8f4-4410-af93-edbd05c1717b
let
	i = 0:.01:49
	
	plot(
		[x->Babylonian(x,N=i) for i=1:5], i,
		label=["Iteration $j" for i=1:1,j=1:5]
	)
	
	plot!(
		sqrt,i,
		c="black",label="sqrt",
		title = "Those Babylonians really knew how to √"
	)
end

# ╔═╡ e4d21c2f-b86b-4566-aa17-0f93dca00e83
with_terminal() do
	@code_native(Babylonian(D((2,1))))
end

# ╔═╡ 790ed2b0-91b1-4b2f-aac7-bab49aea4eb3
map(1:5) do k
	simplify(Babylonian(x_sym,N=k))
end

# ╔═╡ 18cc74a2-ad82-42a2-b2e4-cbde56b1cae3
map(1:5) do k
	simplify(diff(simplify(Babylonian(x_sym,N=k)),x_sym))
end

# ╔═╡ 3b61a005-958c-4c92-bdc1-e9c80189f035
ForwardDiff.derivative(Babylonian, 2)

# ╔═╡ 0e0c20d8-3c41-4ce5-bf31-f951b7761a58
setprecision(3000) do
	round.(Float64.(
			log10.(
				[Babylonian(BigFloat(2),N=k) for k=1:10] .- √BigFloat(2)
			)
		); digits=3)
end

# ╔═╡ ee2a5e81-d540-49de-acaf-1a5831d8838d
let
	x=49
	
	Babylonian(D((x,1))), (√x,.5/√x)
end

# ╔═╡ ae9a2611-cdd0-44d5-9512-0a5a48fed77e
let
	x = π
	
	Babylonian(D((x,1))), (√x,.5/√x)
end

# ╔═╡ 7b3ee0ad-427d-4f78-a90e-db36b85ab1e0
let
	i = .2:.01:49
	plot([x->Babylonian(D((x,1.0)),N=i).f[2] for i=1:5],i)
	plot!(x->.5/√x,i,c="black",label="d(sqrt(x))/dx",
    title = " Babylonians Differentiated")
end

# ╔═╡ cd425623-c338-4b48-92dc-c93bd87dd943
simplify((1 + x_sym) * (1 - x_sym))

# ╔═╡ 077d539a-3c44-4f4d-8e1c-534b9cd693ed
function dBabylonian(x; N = 10) 
    t = (1+x)/2
    t′ = 1/2
    for i = 1:N;  
        t = (t+x/t)/2; 
        t′= (t′+(t-x*t′)/t^2)/2; 
    end    
    t′

end  

# ╔═╡ d49bb920-4d17-4b90-b3bc-eefe0be6476c
let
	x = π
	dBabylonian(x), .5/√x
end

# ╔═╡ 398cb782-1dd9-47fa-bd60-073d2462e93d
# x -  x^3/3! + x^5/5! 
function istanbul_sin(x)
    t = 0.0
    for i=1:10
        t -= (-1.0)^i * (x)^(2*i-1) / prod(1.0:(2*i-1))
    end
    t
end
        

# ╔═╡ ae9e151d-44d8-4d18-9b0d-bc1f2ddb788b
 istanbul_sin(1.0)

# ╔═╡ c9584af1-4c0e-43d6-8975-14f16aeadfcb
istanbul_sin( D((1.0,1.0)))

# ╔═╡ cc48f8e1-fcf8-40c0-8340-4762f8fc42d0
istanbul_sin(D((1.0,1.0)))

# ╔═╡ d9b05e67-445e-4965-9925-5cf2b7ff0422
1/(1+ϵ)  # Exact power series:  1-ϵ+ϵ²-ϵ³-...

# ╔═╡ 424083ab-78f4-4eef-a45d-2a716adbf52f
(1+ϵ)*(1+ϵ)*(1+ϵ)*(1+ϵ)*(1+ϵ)  ## Note this just works (we didn't train powers)!!

# ╔═╡ 5c26b4da-bd33-4bec-b531-3f27f0fe2f09
(1+ϵ)^-10.0

# ╔═╡ e469d433-3002-4a45-abf6-74f608cd6324
(1+ϵ)^(-1)

# ╔═╡ 23a4675e-33cd-4d93-b647-2d6f6c0476ad
function nthroot(x, n=2; t=1, N = 10) 
    for i = 1:N;   t += (x/t^(n-1)-t)/n; end   
    t
end  

# ╔═╡ 21b74252-dce8-46f5-bd06-a4de5877d04c
nthroot(2,3), ∛2 # take a cube root

# ╔═╡ ea52b712-e829-49c7-aecf-b15ef65a5127
nthroot(2+ϵ,3)

# ╔═╡ 6aabb954-c88e-4b96-ac07-58c446fb5cf9
nthroot(7,12), 7^(1/12)

# ╔═╡ c14d1490-483c-469e-9ed1-087c28739743
let
	x = 2.0
	nthroot( x+ϵ,3), ∛x, 1/x^(2/3)/3
end

# ╔═╡ 158dc6f4-50db-4f96-ba8c-94972efadfc3
1/sqrt(8)

# ╔═╡ ac901494-a0aa-404d-a0a3-c04b4fbb1982
z = D((2.0,1.0))

# ╔═╡ 2839a404-308a-45a7-b186-885d2d5e926e
z1 = D1((BigFloat(2.0),BigFloat(1.0)))

# ╔═╡ 9ec45d60-ddb5-495f-ae2d-fe1e3dc8256c
A = randn(3,3)

# ╔═╡ 70fa9a9a-0f2a-4bc7-aad3-b41c059c36d0
x3 = randn(3)

# ╔═╡ a20b1360-33e2-4400-b32e-8611e57bd229
ForwardDiff.gradient(x3->x3'A*x3,x3)

# ╔═╡ d92deac4-0a80-4be4-8648-1b6f78e81aeb
(A+A')*x3

# ╔═╡ aaba684f-e240-47a8-8e87-1c7f0dbcd81c
Strang = let
	n = 4
	SymTridiagonal(2*ones(n),-ones(n-1))
end

# ╔═╡ beac5f59-1b89-42ec-96b7-a200fa19d1e0
Dump(Strang)

# ╔═╡ a0ef4d6f-6db8-45d3-ae34-170bc6cc01cd
Strang\rand(4)

# ╔═╡ bc12b295-0150-47f8-b0f5-5c13c3301ef6
cos(1.0)

# ╔═╡ 2bf3955b-2bec-42b8-81d0-faa162198557
1 / 0

# ╔═╡ b77351e1-eaa1-4df9-8fe8-8e80ed9d2ca9
0 / 0 

# ╔═╡ e7529e7d-6260-4235-acb9-e875f5865877
-1/0

# ╔═╡ ef6b9702-db5d-4eaf-8784-98737289657a
Inf - Inf

# ╔═╡ 5205f6d3-d935-41bc-8c97-dd3dc673dbc5
md"""
IEEE system
"""

# ╔═╡ 1f1876e2-863e-4897-be79-717feb8d0b7d
x = -0.0

# ╔═╡ f05dca86-35d4-4fc7-8b26-aecfca6cad17
Babylonian(D((x,1)))

# ╔═╡ 9394a7af-7b3a-4416-b62e-e28cfada2b94
1/x

# ╔═╡ 28ef5a54-c672-44b2-8745-f97f68d676ef
(0.0 == -0.0)

# ╔═╡ 24aba338-0f4d-4301-a897-96d04fbab344
(1/0.0 == -1/1.0)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[compat]
ForwardDiff = "~0.10.23"
HypertextLiteral = "~0.9.3"
Plots = "~1.23.6"
PlutoTest = "~0.2.0"
PlutoUI = "~0.7.19"
SymPy = "~1.1.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0bc60e3006ad95b4bb7497698dd7c6d649b9bc06"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonEq]]
git-tree-sha1 = "d1beba82ceee6dc0fce8cb6b80bf600bbde66381"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.0"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3287dacf67c3652d3fed09f4c12c187ae4dbb89a"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.4.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "6406b5112809c08b1baa5703ad274e1dded0652f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.23"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "0d185e8c33401084cab546a756b387b15f76720c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.6"

[[PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "92b8ae1eee37c1b8f70d3a8fb6c3f2d81809a1c5"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.0"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "e071adf21e165ea0d904b595544a8e514c8bb42c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.19"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "4ba3651d33ef76e24fef6a598b63ffd1c5e1cd17"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.5"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SymPy]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "PyCall", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "ac560c36cd1c5b836d5aa677ece82bc364b2fa2a"
uuid = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
version = "1.1.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─3abba47d-f2a4-4e8e-a5a4-18aa8724f880
# ╟─c10103e5-23d3-404a-a3a7-d77ca34f9d83
# ╠═713ec357-e015-4b8a-9460-75f611dbe500
# ╟─ea58e432-017d-468c-aa0e-8b119f4846d6
# ╠═850675f8-f111-41fc-931e-02dff4851bfd
# ╠═bc255f04-dac3-4cf5-ad67-d323af9642ed
# ╟─37d2f560-a8f4-4410-af93-edbd05c1717b
# ╟─24af958b-4fe3-40b8-9798-8eabe467eea5
# ╠═d870b113-2250-4363-ab6f-4acab6c3fd55
# ╠═ddba9771-006e-4012-9a7a-4839dd7b6d47
# ╟─f568d9d4-6da8-425e-a0d3-83329ccf72ba
# ╠═07ef3f23-9fdb-406b-88fd-cac39522a18b
# ╠═ee2a5e81-d540-49de-acaf-1a5831d8838d
# ╠═ae9a2611-cdd0-44d5-9512-0a5a48fed77e
# ╠═7b3ee0ad-427d-4f78-a90e-db36b85ab1e0
# ╟─edc5f4de-df01-42d8-9672-ec3c0694bdcd
# ╟─dfa4554a-b09d-4d90-83fb-0b0439231e32
# ╠═e4d21c2f-b86b-4566-aa17-0f93dca00e83
# ╟─73571d42-53a5-4d02-b2e7-cc4a1e42db2c
# ╠═de43006d-f6ad-4e63-af73-d3e7f9a33559
# ╠═faff8119-5d7c-4ae5-84df-ca322505adee
# ╟─89738277-a2ee-4546-902e-4557a1cdc0de
# ╠═cd425623-c338-4b48-92dc-c93bd87dd943
# ╟─460f3ece-5b80-4538-972e-6d7b35ce8195
# ╠═790ed2b0-91b1-4b2f-aac7-bab49aea4eb3
# ╟─09654e11-9e4e-4690-a60b-71f406afa0a4
# ╠═18cc74a2-ad82-42a2-b2e4-cbde56b1cae3
# ╟─6d158dd8-d801-4b17-870e-3103dca2ccb3
# ╠═077d539a-3c44-4f4d-8e1c-534b9cd693ed
# ╠═d49bb920-4d17-4b90-b3bc-eefe0be6476c
# ╠═398cb782-1dd9-47fa-bd60-073d2462e93d
# ╠═ae9e151d-44d8-4d18-9b0d-bc1f2ddb788b
# ╠═e6177b08-5693-4d67-8eb0-bbf029f143aa
# ╠═010e95d3-a206-4d7b-8dcc-37e7cc46c547
# ╠═86f558ed-cbb4-47be-bcd0-1bf10b82a91d
# ╠═c9584af1-4c0e-43d6-8975-14f16aeadfcb
# ╠═cc48f8e1-fcf8-40c0-8340-4762f8fc42d0
# ╠═141df931-1569-43a3-b56e-fd862d181f06
# ╠═25266665-1051-48cd-8c9b-fedb93c1a012
# ╟─7dd8371f-0c1f-43d7-89a7-c4f5244b683b
# ╠═f05dca86-35d4-4fc7-8b26-aecfca6cad17
# ╟─1cdfe1b7-af30-4f9a-ac12-63c682be21cd
# ╟─0760055e-9d68-48be-b3cd-34b72f42ebad
# ╟─46503cf1-95d8-4a9c-98da-c465d66ba0c3
# ╠═91f00d28-d150-468e-88e7-bd67e52d232b
# ╠═ebf49bda-589f-4389-9a2a-cd5382bd0263
# ╠═9baf6cd3-5a63-424a-8576-a1bc5d1b5296
# ╠═c6c0840f-97a5-46ca-9e5c-d711409645ee
# ╠═eef51eb3-f61f-44fb-a5ef-c8092be651bf
# ╠═170439e3-ff0b-4100-86f2-5a920c8645da
# ╠═eaa28814-4ec3-444b-9833-eb57b8a99732
# ╠═45f42fd1-03e8-4901-8cf7-a10d757c9b96
# ╠═433c0a24-3634-4a1b-9f0b-c9137f5f3dfa
# ╠═d9b05e67-445e-4965-9925-5cf2b7ff0422
# ╠═424083ab-78f4-4eef-a45d-2a716adbf52f
# ╠═5c26b4da-bd33-4bec-b531-3f27f0fe2f09
# ╠═e469d433-3002-4a45-abf6-74f608cd6324
# ╟─1c31395e-8c11-4136-af1b-29742834d5d5
# ╠═23a4675e-33cd-4d93-b647-2d6f6c0476ad
# ╠═21b74252-dce8-46f5-bd06-a4de5877d04c
# ╠═ea52b712-e829-49c7-aecf-b15ef65a5127
# ╠═6aabb954-c88e-4b96-ac07-58c446fb5cf9
# ╠═c14d1490-483c-469e-9ed1-087c28739743
# ╟─d5e412cc-8e92-43f5-b28a-87d6b5c14ae0
# ╠═29d26680-646b-4ec7-bd17-0d1b874f7c24
# ╠═91c29413-3e2a-4d1a-99d5-1d9a6565ef25
# ╠═3b61a005-958c-4c92-bdc1-e9c80189f035
# ╠═d07ac980-809d-4fd7-873e-cb97f8bd1915
# ╠═158dc6f4-50db-4f96-ba8c-94972efadfc3
# ╟─33ab0a2c-ab42-489c-b3ea-1ef6382ff35d
# ╠═0e0c20d8-3c41-4ce5-bf31-f951b7761a58
# ╠═e4f63be3-5fa9-47bb-8d55-2dd9b748cd8f
# ╠═ac901494-a0aa-404d-a0a3-c04b4fbb1982
# ╠═2839a404-308a-45a7-b186-885d2d5e926e
# ╠═9ec45d60-ddb5-495f-ae2d-fe1e3dc8256c
# ╠═70fa9a9a-0f2a-4bc7-aad3-b41c059c36d0
# ╠═a20b1360-33e2-4400-b32e-8611e57bd229
# ╠═d92deac4-0a80-4be4-8648-1b6f78e81aeb
# ╠═58ef52f9-d1c8-45dc-b4af-2c1952baa1ad
# ╠═aaba684f-e240-47a8-8e87-1c7f0dbcd81c
# ╠═beac5f59-1b89-42ec-96b7-a200fa19d1e0
# ╠═a0ef4d6f-6db8-45d3-ae34-170bc6cc01cd
# ╠═bc12b295-0150-47f8-b0f5-5c13c3301ef6
# ╠═2bf3955b-2bec-42b8-81d0-faa162198557
# ╠═b77351e1-eaa1-4df9-8fe8-8e80ed9d2ca9
# ╠═e7529e7d-6260-4235-acb9-e875f5865877
# ╠═ef6b9702-db5d-4eaf-8784-98737289657a
# ╟─5205f6d3-d935-41bc-8c97-dd3dc673dbc5
# ╠═1f1876e2-863e-4897-be79-717feb8d0b7d
# ╠═9394a7af-7b3a-4416-b62e-e28cfada2b94
# ╠═28ef5a54-c672-44b2-8745-f97f68d676ef
# ╠═24aba338-0f4d-4301-a897-96d04fbab344
# ╠═1dd7814a-3f20-420b-8367-463646925a0e
# ╠═f911053a-fd18-441d-83bc-82d810c87626
# ╠═39da4330-3507-4ae3-857c-e1a929a678d9
# ╠═ee7767b5-dea4-47f3-9bd6-042967a01948
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
