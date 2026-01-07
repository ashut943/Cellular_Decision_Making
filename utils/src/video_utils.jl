using FFMPEG, FileIO

function create_movie(input_folder::String, output_file::String; frame_rate::Int = 10)
    frames = sort(filter(f -> endswith(f, ".png"), readdir(input_folder; join=true)), by = x -> extract_lambda_value(x))
    
    if isempty(frames)
        println("No frames found in the specified input folder.")
        return
    end
    list_file = joinpath(input_folder, "frames_list.txt")
    open(list_file, "w") do io
        for frame in frames
            println(io, "file '" * abspath(frame) * "'")
        end
    end
    ffmpeg_cmd = `"/opt/homebrew/bin/ffmpeg" -f concat -safe 0 -r $frame_rate -i $list_file -pix_fmt yuv420p -crf 20 -preset medium -y $output_file`
    run(ffmpeg_cmd)
    rm(list_file)
end