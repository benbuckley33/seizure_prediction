structArray = table2struct(mongo_table); 
jsonData = jsonencode(structArray, 'PrettyPrint', true);
% Write to file
file_name = 'mongo_final.json';
fid = fopen(file_name, 'w');
if fid == -1
    error('Unable to open file for writing: %s', file_name)
end
fwrite(fid, jsonData, 'char');
fclose(fid);

disp(['Data saved to ', file_name]);