%load original table and take wanted columns
load('/media/elaX/intanData/ela/individualExperimentDataBase/2025_05_16__08_17_52/2025_05_16__08_17_52_allDataBases.mat')
wanted_columns = [1:12,19,21:24];
updated_db = allDataBases(:,wanted_columns);
%change file location in wanted columns
old_root = '/media/elaX/intanData/ela/channelDataBaseFiles';
new_root = '/media/ben2X/database/raw_channel_data_new';
file_location = strings(height(updated_db),1);
for i = 1:height(updated_db)
    original_path = updated_db.ChannelDataInfo{i}.fileLocation;
    file_location(i) = strrep(original_path, old_root, new_root);
end
updated_db = updated_db(:, [1:12, 14:17]);
updated_db.fileLocation = file_location;
%write as json for mongo
db_struct = table2struct(updated_db);
jfile = jsonencode(db_struct);
fid = fopen('mongo_updated_file.json', 'w');
fprintf(fid,'%s', jfile);
fclose(fid);
