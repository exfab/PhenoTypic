% Execute the pinned Linton calc_vote_stick stage and save every contract field.
repo_root = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))))));
reference_root = fileparts(mfilename('fullpath'));
source_root = fullfile(reference_root, 'source');
fixture_dir = fullfile(repo_root, 'tests', 'fixtures', 'reconnect', 'tensor_voting');
fixture_path = fullfile(fixture_dir, 'linton_calc_vote_stick_r2023b.mat');

addpath(source_root);
set(groot, 'defaultFigureVisible', 'off');
if ~exist(fixture_dir, 'dir')
    mkdir(fixture_dir);
end

response = zeros(11, 13, 'double');
theta = zeros(11, 13, 'double');
response(6, 3) = 1.25;
response(6, 7) = 0.75;
response(6, 11) = 1.5;
response(3, 7) = 0.5;
response(9, 7) = 1.0;
theta(6, 3) = 0.0;
theta(6, 7) = pi / 4;
theta(6, 11) = pi;
theta(3, 7) = pi / 2;
theta(9, 7) = -pi / 2;
sigma = 2.25;

normal_x = -sin(theta);
normal_y = cos(theta);
input_a = response .* normal_x .* normal_x;
input_b = response .* normal_x .* normal_y;
input_d = response .* normal_y .* normal_y;
input_tensor = zeros([size(response), 2, 2], 'double');
input_tensor(:, :, 1, 1) = input_a;
input_tensor(:, :, 1, 2) = input_b;
input_tensor(:, :, 2, 1) = input_b;
input_tensor(:, :, 2, 2) = input_d;

accumulated_tensor = calc_vote_stick(input_tensor, sigma);
[e1, e2, lambda1, lambda2] = convert_tensor_ev(accumulated_tensor);
accumulated_a = accumulated_tensor(:, :, 1, 1);
accumulated_b = accumulated_tensor(:, :, 1, 2);
accumulated_d = accumulated_tensor(:, :, 2, 2);
stick = lambda1 - lambda2;
ball = lambda2;
window_size = floor(ceil(sqrt(-log(0.01) * sigma^2) * 2) / 2) * 2 + 1;
axis_convention = 'array rows down, columns right; theta is Cartesian x-right/y-up axial tangent radians';
source_revision = 'MATLAB Central File Exchange 21051 v1.0.0.0';
source_archive_sha256 = 'c1bb4f14a6b5c8e33e2875a9494b786d0a1463bf2e2169315fef2008f6172148';
runtime = version;

save(fixture_path, 'response', 'theta', 'sigma', 'normal_x', 'normal_y', ...
    'input_a', 'input_b', 'input_d', 'input_tensor', 'accumulated_tensor', ...
    'accumulated_a', 'accumulated_b', 'accumulated_d', 'e1', 'e2', ...
    'lambda1', 'lambda2', 'stick', 'ball', 'window_size', 'axis_convention', ...
    'source_revision', 'source_archive_sha256', 'runtime', '-v7');

rmpath(source_root);
close all force;
