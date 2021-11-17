### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ e1c67d08-4885-4f25-9f3a-d62d9d173c51
using BenchmarkTools

# ╔═╡ 1faf8461-4450-4e8f-a6f4-93cd74b77359
using Libdl

# ╔═╡ c59af4d7-4c35-42f2-b597-a9b6067178a0
using PyCall

# ╔═╡ 05847b84-cebf-4298-a48a-43b214302aff
begin
	using Conda
	Conda.add("numpy")
end

# ╔═╡ 83f68441-0dbf-49ac-892d-34757e17e421
using PlutoTest

# ╔═╡ e99ac763-3f6c-4f6e-b2d5-bf5a04c17da9
using Unitful

# ╔═╡ 5b811217-6d9c-4766-8500-72d2f911c6ec
using DataFrames

# ╔═╡ deed19d0-58f4-4694-9e83-d18a10e706d0
using PlutoUI

# ╔═╡ 3f3cf228-8bf1-486e-b56c-7e5ef9952f22
TableOfContents()

# ╔═╡ f804cccf-73ac-4827-8734-5993914080e7
md"""
# Julia is fast!

Very often, benchmarks are used to compare languages.  These benchmarks can lead to long discussions, first as to exactly what is being benchmarked and secondly what explains the differences.  These simple questions can sometimes get more complicated than you at first might imagine.

The purpose of this notebook is for you to **see a simple benchmark for yourself**.  One can read the notebook and see what happened on the author's Macbook Pro with a 4-core Intel Core I7, or run the notebook yourself.

_(This material began life as a wonderful lecture by Steven Johnson at MIT: [Boxes and registers](https://github.com/mitmath/18S096/blob/master/lectures/lecture1/Boxes-and-registers.ipynb.)).)_


"""

# ╔═╡ 2effeadb-b041-475d-b222-4090c33d9e33
md"""
# `sum`: An easy enough function to understand
"""

# ╔═╡ 38802b3a-b28a-456a-8587-94d8e2fedaf8
md"""
Consider the  **sum** function `sum(a)`, which computes
```math
\mathrm{sum}(a) = \sum_{i=1}^n a_i,
```
where ``n`` is the length of `a`.
"""

# ╔═╡ f5ecd1f2-ae74-4adb-9c43-14d9c38cd64e
a = rand(10^7) # 1D vector of random numbers, uniform on [0,1)

# ╔═╡ 0921f89e-c655-4971-9a81-c6ffa1ad1b4e
sum(a)

# ╔═╡ 0fc44d75-4119-48ae-ae60-cd0936e6f812
md"""
The expected result is `0.5 * 10^7`, since the mean of each entry is `0.5`
"""

# ╔═╡ 9407656b-2c69-4ab2-8513-e1629abc507f


# ╔═╡ 1c59df65-a1f4-41dd-98a1-4e4024b2543c


# ╔═╡ e7b14d05-293c-4839-94b4-3a2f4426e770


# ╔═╡ 96891fb1-e584-46a2-8a82-e9c85fea0d29
md"""
# Benchmarking a few ways in a few languages
"""

# ╔═╡ 96b0be68-6009-4087-8178-29842fc0a7fc
md"""
Julia has a `BenchmarkTools.jl` package for easy and accurate benchmarking:
"""

# ╔═╡ e3271bcf-efbb-4dcd-aa9d-7c22c509ba95
md"""
##  1. The C language

C is often considered the gold standard: difficult on the human, nice for the machine. Getting within a factor of 2 of C is often satisfying. Nonetheless, even within C, there are many kinds of optimizations possible that a naive C writer may or may not get the advantage of.

The current author does not speak C, so he does not read the cell below, but is happy to know that you can put C code in a Julia session, compile it, and run it. Note that the `"\""` wrap a multi-line string.
"""

# ╔═╡ e74940c2-d957-480e-868a-d24d47b83677
C_code = """

	#include <stddef.h>
	double c_sum(size_t n, double *X) {
	    double s = 0.0;
	    for (size_t i = 0; i < n; ++i) {
	        s += X[i];
	    }
	    return s;
	}

	""";

# ╔═╡ bb735d25-fa7a-4662-b6c8-a87cbedfec5f
md"""
#### Compile the C Code
"""

# ╔═╡ 5b31f261-30e9-4277-9a2a-d6b4ceb7efe5
begin
	
	const Clib = tempname()

	cmd = `gcc  -fPIC -O3  -msse3 -xc -shared -o $(Clib * "." * Libdl.dlext) -`
	
	open(cmd, "w") do io
	    print(io, C_code) 
	end
	
end

# ╔═╡ 31d330f8-d52c-4206-b034-75dd303a6d0d
# define a Julia function that calls the C function:
function c_sum(a::Array{Float64})
	ccall(("c_sum", Clib), Float64, (Csize_t, Ptr{Float64}), length(a), a)
end

# ╔═╡ 8ae1f54b-fde1-465a-ba5e-bed2d8c712c9
c_sum(a)

# ╔═╡ fbf5ad29-acd0-42bf-85f2-11ba0b5b2b09
md"""
Let's compare the `c_sum` with the julia built-in `sum`.
"""

# ╔═╡ 317f1826-43dd-469e-955e-0ec6d2a01597
c_sum(a) ≈ sum(a) # type \approx and then <TAB> to get the ≈ symbolb  

# ╔═╡ c2f4827e-df14-49c0-a2f0-4d9aba675192
@test c_sum(a) ≈ sum(a)

# ╔═╡ 01eb38f0-5c62-426c-85cb-c3919afaaf1f
≈  # alias for the `isapprox` function 

# ╔═╡ 70093989-86bb-4cd2-bb48-ec54f12a7372
md"""
_(See the [documentation for `isapprox`](https://docs.julialang.org/en/v1/base/math/#Base.isapprox).)_
"""

# ╔═╡ b9b57c3e-fe70-4e42-9eb2-c29f0932abb7


# ╔═╡ 678f32f2-46e9-4cb1-9e29-cc2f7e83b2b9
md"""
We can now benchmark the C code directly from Julia:
"""

# ╔═╡ 879fd9f7-09cc-49fc-9fc1-9df444722e9e
c_bench = @benchmark $c_sum($a)

# ╔═╡ b8b91f9d-8386-4d25-b135-b3597f5fcc74
md"""
At the [end of this notebook](#results), we create a table called `results`, where we collect the fastest runtimes of our different methods. The first row is our C function:
"""

# ╔═╡ 310d07e0-5cfb-40e9-b17b-5c49853c29f2
md"""
(`ms` means 'millisecond': 1/1000th of a second.)
"""

# ╔═╡ 2d4c9401-fb74-4b5d-be50-fa81478e0af5
md"""
## 2. Python's built in `sum`
"""

# ╔═╡ 3b4b6519-e5e1-4a9d-acfb-cefd5d8ae07e
md"""
The `PyCall` package provides a Julia interface to Python:
"""

# ╔═╡ cae0b598-e587-4439-ae52-d12fa8d81686
# get the Python built-in "sum" function:
pysum = pybuiltin("sum")

# ╔═╡ a2bb1f7c-7bff-4110-8f5e-37ee804b3fd8
@test pysum(a) ≈ sum(a)

# ╔═╡ 1d2ff7f8-7e5c-4ad0-af9c-1f3575d7c80a
py_list_bench = @benchmark $pysum($a)

# ╔═╡ 3c22585e-c949-494d-9731-d0b7dffa1776
results[1:2, :]

# ╔═╡ 1b9a977a-0394-409e-af33-4ab0eb5ce080
md"""
## 3. Python: `numpy` 

> Takes advantage of hardware "SIMD" (but only works when it works)

`numpy` is an optimized C library, callable from Python.
It may be installed within Julia as follows:
"""

# ╔═╡ 4253dc26-32e4-4949-812b-148164adfa75
numpy_sum = pyimport("numpy")["sum"]

# ╔═╡ da217d78-f6a9-4808-9432-e6628fe2f517
py_numpy_bench = @benchmark $numpy_sum($a)

# ╔═╡ 2cfbbb33-4e82-4cf8-9fd1-dc8dc8e0a6a1
@test numpy_sum(a) ≈ sum(a)

# ╔═╡ 782d98e0-2854-4493-bb28-b0e7cdfe603e
md"""
## 4. Python, hand-written
"""

# ╔═╡ 64201e32-0c8e-4057-9c45-3d30add9eecf
sum_py = begin
	py"""
	def py_sum(a):
	    s = 0.0
	    for x in a:
	        s = s + x
	    return s
	"""
	
	py"py_sum"
end

# ╔═╡ e802033c-2327-4295-ae70-5cd69e1e0937
py_hand_bench = @benchmark $sum_py($a)

# ╔═╡ dd2d3797-24cd-4ad6-9248-ccae6146e7ab
@test sum_py(a) ≈ sum(a)

# ╔═╡ 4ce5803b-b5d4-4fb0-86d6-d00a329c03c2
md"""
## 5. Julia (built-in) 

> #### Written directly in Julia, not in C!

You can click on the filename below to jump to Julia's source code.
"""

# ╔═╡ 420c3793-d042-4827-aeef-46a9d5aa551e
@which sum(a)

# ╔═╡ 5b3eb961-e4c7-4d23-8f39-8629e20fc930
jl_bench = @benchmark sum($a)

# ╔═╡ 0fc56904-a192-47a4-92a6-bf4028e0c5ef
md"""
## 6. Julia (hand-written)
"""

# ╔═╡ 6ce188ea-e7d0-49f7-91af-90fc1acac6c5
function mysum(a)   
    s = 0.0  # s = zero(eltype(a))
    for x in a
        s += x
    end
    s
end

# ╔═╡ 5c14ea07-f178-4742-961c-3b8cc9155256
jl_bench_hand = @benchmark $mysum($a)

# ╔═╡ bb1a862f-4772-4e57-90d4-a00d90ce8c7b


# ╔═╡ 96fd66cd-767d-4274-ade6-2a4caad2c6f5


# ╔═╡ bd48bd12-096e-41c7-9278-fd990c765811
md"""
## 7. Julia (hand-written with processor parallelism)
"""

# ╔═╡ f48fcf30-b57a-4960-be2e-d08e5889ed42
function myfastsum(A)   
    s = 0.0  # s = zero(eltype(A))
    @inbounds @simd for a in A  # <-- don't check bounds, parallel on processor
        s += a
    end
    s
end

# ╔═╡ 0dd55c3f-776a-492d-84c4-9f937f98bfd7
j_bench_hand_pp = @benchmark myfastsum($a)

# ╔═╡ 9e9608d8-bd9f-4caf-9879-d09f9468c211
md"""
# Summary

Let's see the result again, sorted by runtime.
"""

# ╔═╡ 034937b4-f557-4bac-8747-f5250084e8a0


# ╔═╡ 6c3e4077-0db5-439a-8490-6429935c3103


# ╔═╡ d848156f-2f93-4aea-878b-d47af9fa22eb


# ╔═╡ a6dc1ca8-f607-42e5-b91c-1608a3e710ee
md"""
#### Appendix
"""

# ╔═╡ 733cc02d-f884-4931-9273-6a4c193b8451
begin

red(x) = HTML("<span style='color: red'>$(x)</span>")
handwritten = red("hand-written")
	
results = DataFrame(
	
	"Method" => [
		md"C" 					
		md"Python - built-in"  	
		md"Python - numpy"  		
		md"Python - $handwritten"
		md"**Julia** - built-in"  
		#md"""**Julia** - $(red("hand-written"))"""
		md"**Julia** - $handwritten"  
		#md"**Julia** - $handwritten parallel"
		md"""**Julia** - $(red("hand-written parallel"))"""
		
	], 
	
	"Best time" => [
		bench === NaN ? NaN : 
		minimum(bench.times) * 1e-6 * u"ms"

		# for bench in [
		# 	c_bench
		# 	py_list_bench
		# 	@isdefined(py_numpy_bench) ? py_numpy_bench : NaN
		# 	py_hand_bench
		# 	jl_bench
		# 	jl_bench_hand
		# 	j_bench_hand_pp
		# ]

		for bench in [
			@isdefined(c_bench) ? c_bench : NaN
			@isdefined(py_list_bench) ? py_list_bench : NaN
			@isdefined(py_numpy_bench) ? py_numpy_bench : NaN
			@isdefined(py_hand_bench) ? py_hand_bench : NaN
			@isdefined(jl_bench) ? jl_bench : NaN
			@isdefined(jl_bench_hand) ? jl_bench_hand : NaN
			@isdefined(j_bench_hand_pp) ? j_bench_hand_pp : NaN
		]
	]
	
);
end

# ╔═╡ cd06669b-97b1-4fb5-ae3e-d28959be59c6
results[1:1, :]

# ╔═╡ 4c4bc734-da8c-4e69-a27e-ffdd545d0b3b
results[1:3, :]

# ╔═╡ 34ea2f58-23b3-4e40-8957-bbd5b0712e77
results[1:4, :]

# ╔═╡ 72f90d42-51c3-4414-8cfb-935fdb8ed906
results[1:5, :]

# ╔═╡ e8b31960-c2f8-495c-b59f-1f434b3d95a8
results[1:6, :]

# ╔═╡ 1544c71f-8ce3-49b2-8570-4bb7eaac7f5f
results

# ╔═╡ de6853b5-0288-4acb-8a82-70ec6c2ceca1
sort(results, "Best time")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Conda = "8f4d0f93-b110-5947-807f-2305c1781a2d"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
BenchmarkTools = "~1.2.0"
Conda = "~1.5.2"
DataFrames = "~1.2.2"
PlutoTest = "~0.2.0"
PlutoUI = "~0.7.19"
PyCall = "~1.92.5"
Unitful = "~1.9.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0bc60e3006ad95b4bb7497698dd7c6d649b9bc06"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "61adeb0823084487000600ef8b1c00cc2474cd47"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

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

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

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

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

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

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

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

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

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

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "4ba3651d33ef76e24fef6a598b63ffd1c5e1cd17"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.5"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "0992ed0c3ef66b0390e5752fe60054e5ff93b908"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.9.2"

[[VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═3f3cf228-8bf1-486e-b56c-7e5ef9952f22
# ╟─f804cccf-73ac-4827-8734-5993914080e7
# ╠═2effeadb-b041-475d-b222-4090c33d9e33
# ╟─38802b3a-b28a-456a-8587-94d8e2fedaf8
# ╠═f5ecd1f2-ae74-4adb-9c43-14d9c38cd64e
# ╠═0921f89e-c655-4971-9a81-c6ffa1ad1b4e
# ╟─0fc44d75-4119-48ae-ae60-cd0936e6f812
# ╟─9407656b-2c69-4ab2-8513-e1629abc507f
# ╟─1c59df65-a1f4-41dd-98a1-4e4024b2543c
# ╟─e7b14d05-293c-4839-94b4-3a2f4426e770
# ╟─96891fb1-e584-46a2-8a82-e9c85fea0d29
# ╟─96b0be68-6009-4087-8178-29842fc0a7fc
# ╠═e1c67d08-4885-4f25-9f3a-d62d9d173c51
# ╟─e3271bcf-efbb-4dcd-aa9d-7c22c509ba95
# ╠═1faf8461-4450-4e8f-a6f4-93cd74b77359
# ╠═e74940c2-d957-480e-868a-d24d47b83677
# ╟─bb735d25-fa7a-4662-b6c8-a87cbedfec5f
# ╠═5b31f261-30e9-4277-9a2a-d6b4ceb7efe5
# ╠═31d330f8-d52c-4206-b034-75dd303a6d0d
# ╠═8ae1f54b-fde1-465a-ba5e-bed2d8c712c9
# ╟─fbf5ad29-acd0-42bf-85f2-11ba0b5b2b09
# ╠═317f1826-43dd-469e-955e-0ec6d2a01597
# ╠═c2f4827e-df14-49c0-a2f0-4d9aba675192
# ╠═01eb38f0-5c62-426c-85cb-c3919afaaf1f
# ╟─70093989-86bb-4cd2-bb48-ec54f12a7372
# ╟─b9b57c3e-fe70-4e42-9eb2-c29f0932abb7
# ╟─678f32f2-46e9-4cb1-9e29-cc2f7e83b2b9
# ╠═879fd9f7-09cc-49fc-9fc1-9df444722e9e
# ╟─b8b91f9d-8386-4d25-b135-b3597f5fcc74
# ╠═cd06669b-97b1-4fb5-ae3e-d28959be59c6
# ╟─310d07e0-5cfb-40e9-b17b-5c49853c29f2
# ╟─2d4c9401-fb74-4b5d-be50-fa81478e0af5
# ╟─3b4b6519-e5e1-4a9d-acfb-cefd5d8ae07e
# ╠═c59af4d7-4c35-42f2-b597-a9b6067178a0
# ╠═cae0b598-e587-4439-ae52-d12fa8d81686
# ╠═a2bb1f7c-7bff-4110-8f5e-37ee804b3fd8
# ╠═1d2ff7f8-7e5c-4ad0-af9c-1f3575d7c80a
# ╟─3c22585e-c949-494d-9731-d0b7dffa1776
# ╟─1b9a977a-0394-409e-af33-4ab0eb5ce080
# ╠═05847b84-cebf-4298-a48a-43b214302aff
# ╠═4253dc26-32e4-4949-812b-148164adfa75
# ╠═da217d78-f6a9-4808-9432-e6628fe2f517
# ╠═2cfbbb33-4e82-4cf8-9fd1-dc8dc8e0a6a1
# ╟─4c4bc734-da8c-4e69-a27e-ffdd545d0b3b
# ╟─782d98e0-2854-4493-bb28-b0e7cdfe603e
# ╠═64201e32-0c8e-4057-9c45-3d30add9eecf
# ╠═e802033c-2327-4295-ae70-5cd69e1e0937
# ╠═dd2d3797-24cd-4ad6-9248-ccae6146e7ab
# ╟─34ea2f58-23b3-4e40-8957-bbd5b0712e77
# ╟─4ce5803b-b5d4-4fb0-86d6-d00a329c03c2
# ╠═420c3793-d042-4827-aeef-46a9d5aa551e
# ╠═5b3eb961-e4c7-4d23-8f39-8629e20fc930
# ╟─72f90d42-51c3-4414-8cfb-935fdb8ed906
# ╟─0fc56904-a192-47a4-92a6-bf4028e0c5ef
# ╠═6ce188ea-e7d0-49f7-91af-90fc1acac6c5
# ╠═5c14ea07-f178-4742-961c-3b8cc9155256
# ╠═e8b31960-c2f8-495c-b59f-1f434b3d95a8
# ╠═bb1a862f-4772-4e57-90d4-a00d90ce8c7b
# ╟─96fd66cd-767d-4274-ade6-2a4caad2c6f5
# ╠═bd48bd12-096e-41c7-9278-fd990c765811
# ╠═f48fcf30-b57a-4960-be2e-d08e5889ed42
# ╠═0dd55c3f-776a-492d-84c4-9f937f98bfd7
# ╠═1544c71f-8ce3-49b2-8570-4bb7eaac7f5f
# ╠═9e9608d8-bd9f-4caf-9879-d09f9468c211
# ╠═de6853b5-0288-4acb-8a82-70ec6c2ceca1
# ╟─034937b4-f557-4bac-8747-f5250084e8a0
# ╟─6c3e4077-0db5-439a-8490-6429935c3103
# ╟─d848156f-2f93-4aea-878b-d47af9fa22eb
# ╟─a6dc1ca8-f607-42e5-b91c-1608a3e710ee
# ╠═83f68441-0dbf-49ac-892d-34757e17e421
# ╠═e99ac763-3f6c-4f6e-b2d5-bf5a04c17da9
# ╠═5b811217-6d9c-4766-8500-72d2f911c6ec
# ╠═733cc02d-f884-4931-9273-6a4c193b8451
# ╠═deed19d0-58f4-4694-9e83-d18a10e706d0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
