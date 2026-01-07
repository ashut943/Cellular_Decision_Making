using Printf

function generate_filename(folder_name, base_name::String)
    return joinpath(folder_name, @sprintf("%s", base_name))
end

function log_to_file(msg)
    println(msg)                 # Print to console
    println(log_file, msg)       # Write to file
end

function extract_lambda_value(filename::String)::Float64
    parts = split(filename, '_')
    idx = findfirst(x -> x == "lambda", parts)
    if idx !== nothing && idx < length(parts)
        try
            return parse(Float64, parts[idx + 1])
        catch
            return Inf
        end
    end
    return Inf
end