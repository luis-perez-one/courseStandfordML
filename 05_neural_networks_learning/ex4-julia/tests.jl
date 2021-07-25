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

macro name(arg)
    string(arg)
end

function print_arr_dims(array::NumericV)::String
    arrayName = @name(array)
    outStr = @sprintf "FYI: %s is a numeric vector " arrayName
    outStr *= @sprintf "with dimensions %i × 1." size(array)[1]
    println(outStr)
end

function print_arr_dims(array::NumericM)::String
    arrayName = @name(array)
    outStr = @sprintf "FYI: %s is a numeric matrix " arrayName
    #outStr = @sprintf "with dimensions %i × %i." size(array)[1] size(array)[2]
    println(outStr)
end

