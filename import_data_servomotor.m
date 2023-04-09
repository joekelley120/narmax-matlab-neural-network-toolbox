function [u, y] = import_data_servomotor()

    data = importdata('data/servomotor/servomotor_0to4V_200Hz.csv');
    
    u = data.data(1:end/2, 6);
    y = data.data(1:end/2, 3);

end