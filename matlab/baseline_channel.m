% IRS-Sim: MATLAB Baseline Channel & 3D Beam Pattern Model
disp('--- MATLAB IRS 3D Beam Pattern Simulation ---');
f = 28e9; % 28 GHz
c = 3e8;
lambda = c/f;

% 1. Create an 8x8 Uniform Rectangular Array (URA) for the IRS
% Element spacing is lambda/2
irs_array = phased.URA('Size',[8 8], 'ElementSpacing', [lambda/2 lambda/2]);

% 2. Calculate the Steering Vector to focus the beam
% Let's steer the beam towards a specific user location
% Azimuth: 30 degrees, Elevation: 10 degrees
steer_angle = [30; 10]; 
steering_vector = phased.SteeringVector('SensorArray',irs_array, 'PropagationSpeed',c);

% The optimal phase shifts to steer the beam are derived from the steering vector
sv = steering_vector(f, steer_angle);
phases = angle(sv); % Extract phases in radians

% 3. Plot the 3D Radiation Pattern
disp('Plotting 3D Radiation Pattern...');
figure;
pattern(irs_array, f, 'PropagationSpeed', c, 'Type', 'directivity', ...
    'CoordinateSystem', 'rectangular', 'Weights', sv);
title('IRS 3D Radiation Pattern (Steered to Az=30, El=10)');

disp('Simulation Complete. You can run this in MATLAB Online.');
