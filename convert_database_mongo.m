load('/media/ela2X/2024_01_23__22_53_33_allDataBases.mat');
old_file_location = cell(size(allDataBases.ChannelDataInfo)); 
for row_idx = 1:length(allDataBases.ChannelDataInfo)
    old_file_location{row_idx} = allDataBases.ChannelDataInfo{row_idx}.fileLocation;
end
new_file_location = "/media/ben2X/database/raw_channel_data";
new_file_location_vector = cell(size(old_file_location));

% Get total number of entries
N = length(old_file_location);  

% Test first 5 indices manually
for i = 1:min(5, N)  % Just test a few cases
    old_file = old_file_location{i}; 
    relative_path = old_file(47:end);
    new_path = strcat(new_file_location, relative_path, "_converted.mat");
    
    fprintf("Index %d: %s\n", i, new_path);  % Print result
end

parpool('local', 10);

parfor row_idx = 1:length(old_file_location)
    old_file = old_file_location{row_idx};
    relative_path = old_file(47:end);
    new_path = strcat(new_file_location, relative_path, "_converted.mat");
    if exist(new_path, 'file') == 2
        new_file_location_vector{row_idx} = new_path;
    else
        new_file_location_vector{row_idx} = NaN;
    end
    if mod(row_idx, 100) == 0
        fprintf('Processed %d of %d rows (%.2f%%)\n', row_idx, length(old_file_location), (row_idx / length(old_file_location)) * 100);
    end
end

kept_columns = setdiff(1:width(allDataBases),13:19);
mongo_table = allDataBases(:,kept_columns);
mongo_table.FileLocation = new_file_location_vector;
save('mongo_file.mat', 'mongo_table', '-v7.3');
delete(gcp('nocreate'));