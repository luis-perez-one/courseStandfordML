∑ = sum
using StatsFuns:logistic
sigmoid = logistic
using CSV:File
using DataFrames
using Printf:@sprintf

# DataTypes
using LinearAlgebra:Transpose
Numeric = Union{Int64,Float64}
NumericV = Union{Array{Int64,1},Array{Float64,1}}
NumericM = Union{Array{Int64,2},Array{Float64,2},Transpose{Float64,Array{Float64,2}},Transpose{Int64,Array{Int64,2}}}
arrayNumericV = Union{Array{Array{Int64,1},1},Array{Array{Float64,1},1}}
arrrayNumericArray = Union{Array{Array{Array{Float64,2},1},1},Array{Array{Array{Int64,2},1},1}}
TupleNumericM = Tuple{NumericM, NumericM};


function maskBiasUnits(array::NumericV)::NumericV
    mask = ones(size(array));
    mask[1,1] = 0;
    maskedArray = array.*mask;
end

function maskBiasUnits(array::NumericM)::NumericM
    mask = ones(size(array));
    mask[:,1] .= 0;
    maskedArray = array.*mask;
end


function powerElements(array::Union{NumericV,NumericM}, power::Int)::NumericM
    # function defined to be broadcasted to elements of a tuple
    array = array.^power
end


function getLabels(y::NumericV)::NumericV
    labels = collect(BitSet(y));
    sort!(labels)
end


function sortArrayByCol(array::Array{Any,2}, col::Int)
    array = DataFrame(array);
    sort!(array, col)
    array = Matrix(array)
end

function getLayersQ(all_θ)::Int
    θ_matricesQ = length(collect(all_θ));
    layersQ = θ_matricesQ+1;
end


function nnClasificationFeedForward(all_θ, X, y, labels, layersQ::Int=getLayersQ(all_θ))
    labelsQ = length(labels);
    hx_and_indices = reshape([],0,labelsQ+1);
    #zDict = Dict(i => Dict(j => Array{Float64,2}  for j in 1:layersQ) for i in 1:labelsQ);
    all_z = [];
    for lb in labels
        push!(all_z, [])
        ly = 1;
        lb_idx = findall(==(lb), y);
        a_lb_ly = X[lb_idx,:];
        ly += 1;
        while ly ≤ layersQ
            # add bias units to a_lb_ly
            a_lb_ly = [ones(size(a_lb_ly)[1],1) a_lb_ly];
            a_previous_ly = a_lb_ly;
            θ_previous_ly = all_θ[ly-1];
            z_lb_ly = a_previous_ly*θ_previous_ly';
            push!(all_z[lb],z_lb_ly);
            a_lb_ly = sigmoid.(z_lb_ly);
            ly += 1;
        end
        ly = 1;
        hx_lb = a_lb_ly;
        hx_and_indices_lb = hcat(hx_lb, lb_idx);
        hx_and_indices = vcat(hx_and_indices, hx_and_indices_lb);
    end
    hx_and_indices = sortArrayByCol(hx_and_indices, labelsQ+1);
    hx = hx_and_indices[:,1:labelsQ];
    hx = convert(Array{Float64,2}, hx)
    return hx, all_z
end


function nnCostClasification(all_θ::TupleNumericM, hx, y::NumericV, labels::NumericV, λ::Numeric)::Numeric
    J = 0;
    for lb in labels
        lb_idx = findall(==(lb), y)
        hx_lb = hx[lb_idx,:];
        y_lb = vec(zeros(size(labels))); y_lb[lb] = 1;
        J += ∑(-((log.(hx_lb))*y_lb)-(log.(1 .- hx_lb)*(1 .- y_lb)));
    end
    casesQ = size(hx)[1];
    J = (1/casesQ)*J;
    # add regularization cost
    J += (λ/(2*casesQ))* ∑(∑.(powerElements.(maskBiasUnits.(all_θ),2)));
end


function calcSigmoidDerivate(z::Numeric)::Numeric
    sigmoid(z)*(1-sigmoid(z));
end

function calcSigmoidDerivate(z::NumericM)::NumericM
    sigmoid.(z).*(1 .- sigmoid.(z));
end

function calcSigmoidDerivate(z::Array{Any,1})
    calcSigmoidDerivate.(z);
end


function calcNiceInitϵ(layerInputsQ::Int, layerOutputsQ::Int)::Numeric
    initϵ = √6/√(layerInputsQ+layerOutputsQ)
end


function genRandomInitialθ(layerInputsQ::Int, layerOutputsQ::Int, initϵ::Numeric=calcNiceInitϵ(layerInputsQ,layerOutputsQ))::NumericM
    initϵInt = floor(initϵ*100);
    randInitialθ = rand(-initϵInt:initϵInt,layerOutputsQ,layerInputsQ+1);
    randInitialθ = randInitialθ./100;
end


function nnClasificationBackPropagation(hx::NumericM, all_θ::TupleNumericM, all_SigmoidDerivates::arrrayNumericArray, layersQ::Int, y::NumericV, labels::NumericV)
    all_δ = [];
    labelsQ = length(labels);
    casesQ = size(hx)[1];
    ly = layersQ;
    lb_idx_v = collect(1:labelsQ);
    outputLayer = map(idx_lb -> (vec(hx[:,idx_lb])), lb_idx_v); #Matrix of subarrays where each of them correspond to each row of hx'
    δ_ly =  map(idx_lb -> outputLayer[idx_lb] .- y, lb_idx_v);
    push!(all_δ, δ_ly)
    ly -= 1;
    while ly ≥ 2
        δ_previous_ly = δ_ly;
        θ_ly = all_θ[ly][:,2:size(all_θ[ly])[2]]; 
        δ_ly = map(idx_lb -> (δ_previous_ly[idx_lb][findall(==(idx_lb), y)]*(θ_ly[idx_lb,:])').*all_SigmoidDerivates[idx_lb][ly-1], lb_idx_v);
        # note, there is one matrix less than the total number of layers in the all_SigmoidDerivates[idx_lb] array, hence the [ly-1] slicing
        push!(all_δ, δ_ly)
        ly -= 1;
    end
    reverse!(all_δ)
end




#parameters
# csv files -> DataFrames -> arrays
X =  Matrix{Float64}(File("./X.csv", header=false)
    |> DataFrame);
y = vec(Matrix(File("./y.csv", header=false) |> DataFrame));
θ1 = Matrix{Float64}(File("./theta1.csv", header=false)
    |> DataFrame);
θ2 = Matrix{Float64}(File("./theta2.csv", header=false) |> DataFrame);
λ = 1;

#implementation
all_θ = (θ1, θ2);
labels = getLabels(y);
layersQ = getLayersQ(all_θ);
hx, all_z = nnClasificationFeedForward(all_θ, X, y, labels, layersQ);
J = nnCostClasification(all_θ, hx, y, labels, λ);
all_SigmoidDerivates= calcSigmoidDerivate.(all_z);
all_δ = nnClasificationBackPropagation(hx, all_θ, all_SigmoidDerivates, layersQ, y, labels);


## consider next

# collect(Iterators.flatten(all_δ[1]))
# check that 125000/25 = 5000   
    # 25 is the output nodes from layer 1

# collect(Iterators.flatten(all_δ[2]))
# check that 50000/10 = 5000
    # 10 is the output nodes from layer 2

# what is what?
    # how to slice / reshape? 
        # check original output from all_δ[1] and all_δ[2]
