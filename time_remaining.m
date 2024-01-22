clc; clear; close all;

dirPath = '/Users/g.park/Library/CloudStorage/GoogleDrive-gspark@kentech.ac.kr/공유 드라이브/BSL-Data/Processed_data/Hyundai_dataset/AgingDOE_cycle2';

% 디렉토리 존재 확인
if ~isfolder(dirPath)
    error('폴더가 존재하지 않습니다: %s', dirPath);
end

% .mat 파일의 목록을 얻습니다.
fileList = dir(fullfile(dirPath, '*.mat'));

% 디렉토리 내 .mat 파일이 있는지 확인
if isempty(fileList)
    warning('폴더 내에 .mat 파일이 없습니다: %s', dirPath);
else
    % cycle 변수 미리 정의
    % cycle = zeros(1, length(fileList));
    
    % .mat 파일들을 순회하면서 작업 수행
    for i = 1:length(fileList)
        currentFile = fullfile(dirPath, fileList(i).name);
        loaddata = load(currentFile);
        
        % 스트럭트에서 cycle 필드의 마지막 값을 추출
          cycle(i) = loaddata.data(end,end);
        
        % 여기에 추가 작업을 수행하세요
    end
    
    % 마지막 값을 출력 또는 필요한 작업 수행
    disp('마지막 cycle 값들:');

end

%     % cycle 배열에서 마지막 값 추출
%     lastCycleValue = cycle(end);
% 
%     % 마지막 값 출력 또는 필요한 작업 수행
%     disp('마지막 cycle 값:');
%     disp(lastCycleValue);
% end