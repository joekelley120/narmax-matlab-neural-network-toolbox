function [u, y] = import_data_maglev()

    u = [];
    y = [];
    for i = 5: 15
        data = importdata(sprintf('data/maglev/Data_Train_Ex%s.mat', num2str(i)));
        u = [u; data.Voltage'];
        y = [y; data.Mag_Pos_F'];
    end

end