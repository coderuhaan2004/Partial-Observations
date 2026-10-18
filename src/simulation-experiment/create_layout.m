function layout = create_layout()
%CREATE_LAYOUT Fixed positions in metres for the reproducible 3x3 experiment.

layout.su_positions_m = [2, 3; 10, 4; 5, 12];
layout.pu_positions_m = [0, 0; 8, 0; 4, 8];
layout.reference_distance_m = 1;
layout.path_loss_exponent = 3.71;
layout.beta = 10^(-3.154);
layout.shadow_sigma_db = 3.65;
end
