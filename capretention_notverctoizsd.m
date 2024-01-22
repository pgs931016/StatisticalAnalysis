clc; clear; close all;

dirPath = '/Users/g.park/Library/CloudStorage/GoogleDrive-gspark@kentech.ac.kr/공유 드라이브/Battery Software Lab/Processed_data/Hyundai_dataset/4CPD_1C_231212';
fileList = dir(fullfile(dirPath, '*.mat'));

aging_data = cell(length(fileList), 1);

% 각 파일별로 Q, I, cycle 값을 저장할 셀 배열 초기화
allDataCell = cell(length(fileList), 1);

% 파일별로 다른 색상을 사용하기 위한 색상 벡터
colors = hsv(length(fileList));

% 사용자가 직접 입력하는 온도 정보
temperatures = [25, 10, -10]; % 계속해서 온도를 추가하십시오.

% i를 두 개씩 묶어서 플로팅
for i = 1:2:length(fileList)
    figure;
    
    for j = i:i+1
        filePath = fullfile(dirPath, fileList(j).name);
        aging_data{j} = load(filePath);

        % 사용자가 직접 입력한 온도 정보 사용
        temperatureInfo = temperatures(round(j/2));
        
        % 'D'에 해당하는 인덱스 찾기
        indices = find(strcmp({aging_data{j}.data.type}, 'D'));

        % 해당 인덱스에 해당하는 값을 가져와서 셀 배열에 저장
        Dcap = aging_data{j}.data(indices);

        % 파일별로 구분해서 Q, I, cycle 값을 저장
        dataStruct = struct('Q', {}, 'I', {}, 'cycle', {});
        for k = 1:length(Dcap)
            Q = abs(trapz(Dcap(k).t, Dcap(k).I)) / 3600;
            I = Dcap(k).I;
            cycle = Dcap(k).cycle;

            % 구조체에 Q, I, cycle 값을 저장
            dataStruct(end+1) = struct('Q', Q, 'I', I, 'cycle', cycle);
        end

        % 색상을 지정하여 플로팅
        for k = 1:length(dataStruct)
            scatter(dataStruct(k).cycle, ((dataStruct(k).Q-dataStruct(1).Q)/dataStruct(1).Q)*100, 'MarkerFaceColor', colors(j, :), 'MarkerEdgeColor', colors(j, :));
            hold on;
        end

        % 셀 배열에 저장
        allDataCell{j} = dataStruct;
    end

    % 온도 정보를 이용하여 타이틀 설정
    title([' ', num2str(temperatureInfo), ' degC']);
    
    xlabel('Cycle / n');
    ylabel('Cap / mAh');
    ylim ([-20 20]);
    customsettings;
end
