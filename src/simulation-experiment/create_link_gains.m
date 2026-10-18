function links = create_link_gains(layout, su, pu)
%CREATE_LINK_GAINS Apply the paper's distance/log-normal path-loss model.
%
% |h|^2 = beta * (d0/d)^alpha * 10^(-psi/10), psi ~ N(0, 3.65^2).
% Shadowing is sampled once per physical PU-SU link, not once per frame.

links = repmat(struct("pu_index", 0, "distance_m", 0, ...
    "shadow_db", 0, "power_gain", 0, "complex_gain", 0), ...
    numel(su), numel(pu));
for suIndex = 1:numel(su)
    for puIndex = 1:numel(pu)
        distance = max(norm(layout.su_positions_m(suIndex, :) - ...
            layout.pu_positions_m(puIndex, :)), layout.reference_distance_m);
        shadowDb = layout.shadow_sigma_db * randn();
        powerGain = layout.beta * ...
            (layout.reference_distance_m / distance)^layout.path_loss_exponent * ...
            10^(-shadowDb / 10);
        phase = 2 * pi * rand();
        links(suIndex, puIndex).pu_index = puIndex;
        links(suIndex, puIndex).distance_m = distance;
        links(suIndex, puIndex).shadow_db = shadowDb;
        links(suIndex, puIndex).power_gain = powerGain;
        links(suIndex, puIndex).complex_gain = sqrt(powerGain) * exp(1i * phase);
    end
end
end
