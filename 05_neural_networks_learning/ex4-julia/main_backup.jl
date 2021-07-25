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
    for lb in labels
        ly = 1;
        lb_idx = findall(==(lb), y)
        a_ly = X[lb_idx,:];
        ly += 1;
        while ly ≤ layersQ
            # add bias units to a_ly_lb
            a_ly = [ones(size(a_ly)[1],1) a_ly];
            a_previous_ly = a_ly;
            θ_previous_ly = all_θ[ly-1];
            z_ly = a_previous_ly*θ_previous_ly';
            a_ly = sigmoid.(z_ly);
            ly += 1;
        end
        ly = 1;
        hx_lb = a_ly;
        hx_and_indices_lb = hcat(hx_lb, lb_idx);
        hx_and_indices = vcat(hx_and_indices, hx_and_indices_lb);
    end
    hx_and_indices = sortArrayByCol(hx_and_indices, labelsQ+1);
    hx = hx_and_indices[:,1:labelsQ];
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

calcSigmoidGradient = calcSigmoidDerivate;


function calcNiceInitϵ(layerInputsQ::Int, layerOutputsQ::Int)::Numeric
    initϵ = √6/√(layerInputsQ+layerOutputsQ)
end


function genRandomInitialθ(layerInputsQ::Int, layerOutputsQ::Int, initϵ::Numeric=calcNiceInitϵ(layerInputsQ,layerOutputsQ))::NumericM
    initϵInt = floor(initϵ*100);
    randInitialθ = rand(-initϵInt:initϵInt,layerOutputsQ,layerInputsQ+1);
    randInitialθ = randInitialθ./100;
end

function substractElements(sub_array::Array{Any,1},vector::NumericV)
    sub_array .- vectors
end


function gen_initial_z(all_θ, labelsQ::Int)
    initial_z_singleLy = [];
    for θ in all_θ
        initial_z_singleLy = [initial_z_singleLy, ones(size(θ))];
    end
    initial_z_singleLy[1]=initial_z_singleLy[1][2];
    
    initial_z = [];
    for lb in labelsQ
        initial_z = [initial_z, initial_z_singleLy];
    end
    initial_z = initial_z[2]
end


function nnClasificationBackPropagation(hx, all_θ, layersQ::Int, y, labels)
    # calculate the gradients -i.e. partial derivates with respect to θ- of the cost (J)
    nodesQ = labelsQ = length(labels);
    casesQ = size(hx)[1];
    ly = layersQ;
    idx_v = collect(1:labelsQ);
    outputLayer = map(idx -> (vec(hx[:,idx])), idx_v); #Matrix of subarrays where each of them correspond to each row of hx'
    δ_ly =  map(idx -> allCasesOutputLayer[idx] .- y, idx_v);
    ly -= 1;
    while ly ≥ 2
        δ_previous_ly = δ_ly;
        θ_ly = all_θ[ly];
        z_ly = 



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
hx = nnClasificationFeedForward(all_θ, X, y, labels, layersQ);
J = nnCostClasification(all_θ, hx, y, labels, λ);
allCasesδ_ly = nnClasificationBackPropagation(hx, layersQ, y, labels);