function convert_matlab_function(input_file, output_file)
    data = load(input_file);
    converted_data = table2struct(data.theChannelData.chanData, "ToScalar",true);
    save(output_file, '-struct', 'converted_data');
end
    
  
