% MRS 2023
% spatiotemporal mixture of gaussian processes
% Created by Siva Kailas
% Date: 07/24/2023

% with distributed control and consensus density function learning

function [rms_stack, var_stack, cf, max_mis, model, pred_h, pred_Var] = main_bot_distribute(varargin)
%debug variable
failures = [];
%choose starting timestep to use from air temp dataset
sweep = 0;

%length of timestep in dataset (for air data, each index is 1 month)
periodic_step = 1;

% for sweep = 1:2:894-6
    %sweep = 334

%define global vars
global num_gau num_bot Xss Fss eta

load('sample_data_stalk_count.mat'); % read stalk count data (not actually using the stalk count dataset, but we need the miscellaneous variables such as the map)

%define the 4-neigborhood graph in the env to do path planning over
A = latticeAdjacencyMatrix(45,21);
Agraph = graph(A);
%uncomment if you want to plot, only for debug
%plot(Agraph, 'Layout', 'force')

%total length of simulation
total_sim_timesteps = 20;


%read in air temperature data
air = ncread("air.sig995.mon.mean.nc", "air");

Xss_base = Xss; %base map coords (no timestep)
% Fss_base = Fss;
Fss = []; %empty env values, 945n x 1 vec where n = num timesteps
Xss_big = []; %empty map coords + timestep values (proxy var used to create Xss)

%get min and max air temps in data
min_air = min(air(1:21, 1:45, :),[],"all");
max_air = max(air(1:21, 1:45, :),[],"all");

%iterate over dataset and get values over bounded env
for times = 1:periodic_step:total_sim_timesteps
    air_x_y_1 = air(1:21, 1:45, times + sweep);
    air_x_y_1 = air_x_y_1 - min_air;
    Fss = [Fss; reshape(air_x_y_1, [], 1)]; %reshape into vector to add to Fss
    Xss_temp = [Xss ones(size(Xss, 1), 1)*(times)]; %add timestep to Xss_base and add to Xss_temp
    Xss_big = [Xss_big; Xss_temp]; %add Xss_temp to Xss_big
end
%set Xss to Xss_big and Fss_big = Fss
Xss = Xss_big;
Fss_big = Fss;

% if you wish to set GP hyperparams, do so here and remove def_hyp2 = []
def_hyp2.mean = [];
def_hyp2.cov = [2 2];
def_hyp2.lik = -10;

% to use bayesian optimization for GP hyperparams, set def_hyp2 = []
def_hyp2 = [];

% set of user-defined hyperparams (last param = value to set)
parser = inputParser;
addOptional(parser, 'algo', 'gmm'); % how to select break-off swarm
addOptional(parser, 'bots', []);
addOptional(parser, 'num_gau', 3); %number of gaussian components
addOptional(parser, 'beta', 1); %exploration hyperparam for UCB-based heuristic
addOptional(parser, 'unit_sam', 300); %number of initial samples for bots
% addOptional(parser, 'eta', 0.1);
addOptional(parser, 'g_num', 3); %number of bots
addOptional(parser, 'hyp2_new', def_hyp2); %GP hyperparams
addOptional(parser, 'kp', 1); %robot gain (not used since we have more formal constraints in IPP now)
addOptional(parser, 'it_num', total_sim_timesteps); %number of sim iterations to do
addOptional(parser, 'save_flag', false); %ignore, not used
%For zeroing out mean in UCB
addOptional(parser, 'kappa', 0);

parse(parser, varargin{:});

algo = parser.Results.algo;
num_gau = parser.Results.num_gau;
beta = parser.Results.beta;
unit_sam = parser.Results.unit_sam;
% eta = parser.Results.eta;
g_num = parser.Results.g_num;
bots = parser.Results.bots;
hyp2_new = parser.Results.hyp2_new;
kp = parser.Results.kp;
it_num = parser.Results.it_num;
save_flag = parser.Results.save_flag;
%Zeroing out mean if needed for UCB
kappa = parser.Results.kappa;

% gamma_t = 0;
%Mutual Info lower bound for GPs
gamma_gp = zeros(g_num, 1);


%%%%%%%%%%%%%%%%%%% dataset dependent variables
%dynamically adjust GMM components if needed
while true
    map_x = 20;
    map_y = 44;
    map_z = [min_air - min_air,max_air - min_air];
    
    num_init_sam = unit_sam*num_gau + 1;%10; % initial number of samples for each robot
    
    k = g_num; % number of robots
    num_bot = g_num;
    stop_flag = 0;
    %%%%%%%%%%%%%%%%%%%
    
    %boolean flags for type of IPP + waypoint selection
    useApproxMI = true; %use secretary alg for waypoint sel if true else naive max
    max_path_length = 5*1; %set max allowable path length
    useRatioPath = false; %use ratio-based heuristic for path selection if true else use direct info
    
    rng('shuffle')
    g=[];
    
    [label_rss, model_rss, llh_rss] = mixGaussEm_gmm(Fss', num_gau); % centralized GMM version
    
    if size(unique(label_rss),2)~=num_gau
        %error('reduce num_gau!');
        warning('reduce num_gau!');
        num_gau = num_gau - 1;
    else
        break
    end

end

%random sample from GMM for initial data for robots
pilot_Xs_stack = zeros(unit_sam*num_gau,3,num_bot);

for ikoo = 1:num_bot
    ind_ttmp = zeros(unit_sam,num_gau);
    
    for kik = 1:num_gau
        sap_tmp = find(label_rss==kik);
        ind_ttmp(:,kik) = sap_tmp(randperm(length(sap_tmp),unit_sam));
    end
    
    ind_ttmp = ind_ttmp(:);
    
    % pilot_Xs = [rand(num_init_sam,1)*map_x rand(num_init_sam,1)*map_y];
    pilot_Xs_stack(:,:,ikoo) = Xss(ind_ttmp,:);
    
end

%%%%%%% initialization
if isempty(bots)  %nargin < 1

    init_abg = zeros(1,num_gau);
    tmp_init = cell(1,num_bot);
    [tmp_init{:}] = deal(init_abg);
    bots = struct('alpha_K',tmp_init,'belta_K',...
        tmp_init,'gamma_K',tmp_init,'self_alpha',tmp_init,'self_belta',tmp_init,'self_gamma',tmp_init,...
        'dot_alpha_K',tmp_init,'dot_belta_K',tmp_init,'dot_gamma_K',tmp_init,'Nm_ind',[]);
    
    packets = struct('alpha_K',[],'belta_K',[],'gamma_K',[]);
    
    
    if numel(unique(label_rss))~= num_gau
        warning('unexpected label_rss');
        return;
    end
    
    for ijk = 1:k
        bots(ijk).Xs = pilot_Xs_stack(:,:,ijk); %[rand(num_init_sam,1)*map_x rand(num_init_sam,1)*map_y];  %[rand*map_x rand*map_y];  %  starting points for the robots, can be set of points from pilot survey
        
        bots(ijk).Xs(end+1,:) = [rand*map_x rand*map_y, 1]; % replace by starting points in a smaller area
        
        bots(ijk).Nm_ind = get_idx(bots(ijk).Xs, Xss);  % initial index for each robot
        g(ijk,:) = bots(ijk).Xs(end,:); % in coverage control, specify generator's positions (starting positions)
        bots(ijk).Fs = Fss(bots(ijk).Nm_ind);
    end
    % seperate into two loops since we want separate control of rng,
    % otherwise the above commands will  generate same robot positions.
    
    for ijk = 1:k
        [~, model, ~, ~, ~] = mixGaussEm_gmm(Fss(bots(ijk).Nm_ind)', num_gau); % initialize: mu, Sigma, alpha
        Nm = length(bots(ijk).Nm_ind);
        bots(ijk).mu_K = model.mu;
        bots(ijk).Sigma_K = model.Sigma;
        bots(ijk).self_alpha = model.w;
        %         bots(ijk).alpha_K = model.w;
        
        [~, alpha_mnk] = mixGaussPred_gmm(Fss(bots(ijk).Nm_ind)', model); % get alpha_mnk:   Nm x num_gau
        self_alpha = sum(alpha_mnk,1); %./Nm;  % 1 x num_gau    Nm*alpha_mk
        y_mn = Fss(bots(ijk).Nm_ind);    %   Nm x 1
        bots(ijk).belta_K = sum(alpha_mnk.*y_mn,1);  %  1 x num_gau  belta_mk
        % mu_K = self_belta./(Nm*self_alpha);  % 1 x num_gau   mu_mk
        bots(ijk).gamma_K = sum(((repmat(y_mn,[1,num_gau])-model.mu).^2).*alpha_mnk, 1);  %  1 x num_gau
        bots(ijk).alpha_K = self_alpha;
        
        for ijk_rec = 1:num_bot
            bots(ijk).packets(ijk_rec) = packets;   % initialize packets struct for every robots
        end
        if ijk ~= num_bot && ijk~=1
            bots(ijk).neighbor = [ijk+1, ijk-1];%setdiff(1:num_bot, ijk);    % find(adj_A(ijk,:)>0);  % get neighbor id, intialize from a fully connected graph
        elseif ijk == num_bot
            bots(ijk).neighbor = ijk - 1;
        elseif ijk == 1
            bots(ijk).neighbor = ijk + 1;
        end
    end
    
    for ijk = 1:num_bot     % reset packets to the bot themselves
        packets.alpha_K = bots(ijk).alpha_K;
        packets.belta_K = bots(ijk).belta_K;
        packets.gamma_K = bots(ijk).gamma_K;
        bots(ijk).packets(ijk) = packets;
    end
    
    
else
    g_num = length(bots);
    k = g_num;
end
g = g';


%% initiate swarm status

s_num = length(Fss); %1e6;
s = Xss';



%g = randi([1 599],2,10)/100;
%
%  Carry out the iteration.
%
step = 1 : it_num;
e = nan ( it_num, g_num );
gm = nan ( it_num, 1 );
cf = nan(it_num, 1);
max_mis = nan(it_num, 1);
rms_stack = nan(it_num, 1);
var_stack = nan(it_num, 1);
avg_rms_stack = nan(it_num, 1);
avg_max_mis = nan(it_num, 1);

%%  initialize for consensus loop
max_iter = 1000; % consensus communication round

% first round communication
for ijk = 1:num_bot
    bots = transmitPacket(bots, ijk); % update communication packets
end

hist_alpha_K = zeros(max_iter,num_gau,num_bot);
hist_belta_K = zeros(max_iter,num_gau,num_bot);
hist_gamma_K = zeros(max_iter,num_gau,num_bot);

hist_alpha_K_norm = zeros(max_iter,num_gau,num_bot);
hist_belta_K_norm = zeros(max_iter,num_gau,num_bot);
hist_gamma_K_norm = zeros(max_iter,num_gau,num_bot);

hist_mu_K_norm = zeros(max_iter,num_gau,num_bot);

true_alpha_K = repmat(model_rss.w,[max_iter 1]);

indices = [];
indices_exact = [];

for it = 1 : it_num
    it    % coverage control iteration step
    %for ten_samps = 1:10
    loop_flag = true;
    cur_iter = 1;

%     Fss_t_1 = Fss;
%     s_num_t_1 = s_num;
    % retain collected samples as anchor points for GP regression
    temp_Fss = [];
    temp_Xss = [];
    for ijk = 1:num_bot
        Nm = length(bots(ijk).Nm_ind);
        for dijk = 1:Nm
            if size(temp_Xss, 1) == 0 || ismember(Xss(bots(ijk).Nm_ind(dijk), :), temp_Xss, 'rows') == 0
                temp_Fss = [temp_Fss; Fss(bots(ijk).Nm_ind(dijk))];
                temp_Xss = [temp_Xss; Xss(bots(ijk).Nm_ind(dijk), :)];
                bots(ijk).Nm_ind(dijk) = size(temp_Fss, 1);
            else
                [Lia, Locb] = ismember(Xss(bots(ijk).Nm_ind(dijk), :), temp_Xss, 'rows');
                bots(ijk).Nm_ind(dijk) = Locb;
            end
        end
    end
    Fss = temp_Fss;
    Xss = temp_Xss;

    % move env to next timestep
    for temp_it = it-0:it+0
        if temp_it > 0
            t = ones(size(Xss_base, 1), 1)*temp_it;
            Xss_t = [Xss_base t];
            Xss = [temp_Xss; Xss_t];
            Fss = [temp_Fss; Fss_big(945*(it-1)+1:945*it)];
        end
    end

    s_num = length(Fss); %1e6;
    s = Xss';

    % after consensus loop, updated variable:  1) bots.neighbor, 2) new
    % local model, 3) bots.Nm_ind (after execution)
    
    %% begin consensus process to refine local model for each robot
    
    if ~stop_flag
        while loop_flag  % for first round, use neighbors defined by default from the part of initialization above
            
            for ijk = 1:num_bot
                %             bots(ijk).neighbor = find(adj_A(ijk,:)>0);   % confirm neighbors
                bots = transmitPacket(bots, ijk); % update communication packets
                bots = updateBotComputations(bots, ijk);  % then update self and consensus
            end
            for dijk = 1:num_bot
                hist_alpha_K(cur_iter,:,dijk,it) = bots(dijk).alpha_K;  % record bot 1's estimate of alpha_K
                hist_belta_K(cur_iter,:,dijk,it) = bots(dijk).belta_K;
                hist_gamma_K(cur_iter,:,dijk,it) = bots(dijk).gamma_K;
                
                hist_alpha_K_norm(cur_iter,:,dijk,it) = norm_prob(bots(dijk).alpha_K);  % record bot 1's estimate of alpha_K
                hist_belta_K_norm(cur_iter,:,dijk,it) = norm_prob(bots(dijk).belta_K);
                hist_gamma_K_norm(cur_iter,:,dijk,it) = norm_prob(bots(dijk).gamma_K);
                
                hist_mu_K_norm(cur_iter,:,dijk,it) = bots(dijk).mu_K;
            end
            cur_iter = cur_iter+1;
            if cur_iter > max_iter
                cur_iter = cur_iter - 1;
                %              figure;plot(1:cur_iter, hist_alpha_K_norm(:,1,1)); % plot converging profile for robot 1 w.r.t. alpha_1
                break;
            end
            %     figure(1);
            %     hold on;
            %     plot(1:cur_iter, true_alpha_K(1:cur_iter,1),1:cur_iter, hist_alpha_K(1:cur_iter,1));
        end   % end of consensus part
    end
    
    
    %     ind_current = vertcat(model.ind);
    %     [mu_gpml, s2_gpml, hyp_gpml, rms_gpml] = gpml_rms(ind_current,Xss,Fss,Xss,Fss);
    
    %
    %  Compute the Delaunay triangle information T for the current nodes.
    %
    t = delaunay(g(1,:),g(2,:));
    
    %  For each sample point, find K, the index of the nearest generator.
    %  We do this efficiently by using the Delaunay information with
    %  Matlab's DSEARCH command, rather than a brute force nearest neighbor
    %  computation.
    %
    k = powercellidx (g(1,:),g(2,:),s(1,:),s(2,:)); % for each point, label it by nearest neighbor  wts k - num_points x 1
    pred_h = zeros(length(Fss),1);
    pred_Var = zeros(length(Fss),1);
    %     pred_rms = zeros(length(num_bot),1);
    pred_post = zeros(length(Fss),length(Fss));
    
    Nm_inds_all_bots = [];
    for iij = 1:num_bot
        %         idx_tmp = find(k==iij); % get corresponding index of the monitored points by bot iij
        %         bots(iij).Nm_ind = bots(iij).Nm_ind(:)';
        [pred_h(k==iij), pred_Var(k==iij), ~, post] = gmm_pred_wafr(Xss(k==iij,:), Fss(k==iij), bots(iij), 'hyp2_new', hyp2_new);
        Nm_inds_all_bots = [Nm_inds_all_bots; reshape(unique(bots(iij).Nm_ind), [], 1)];
%         % uncomment below for sparse cov approx calc for MGPs
%         post_idxs = (k == iij);
%         curr_row_post = 1;
%         curr_col_post = 1;
%         for pq = 1:length(k==iij)
%             if post_idxs(pq) == 1
%                 curr_col_post = 1;
%                 for rs = 1:length(k==iij)
%                     if post_idxs(rs) == 1
%                         pred_post(pq, rs) = pred_post(pq, rs) + post(curr_row_post, curr_col_post);
%                         curr_col_post = curr_col_post + 1;
%                     end
%                 end
%                 curr_row_post = curr_row_post + 1;
%             end
%         end
    end
    % unimodel GP regression
    % used for submodular function optimization for MI approximation
    [Xss_C, Xss_ia, ~] = unique(Xss(Xss(:,3) == it, :), 'rows', 'sorted');
    Fss_C = Fss(Xss(:,3) == it);
    [~, ~, ~, ~, pred_post] = gpml_rms([],Xss(Nm_inds_all_bots,:),Fss(Nm_inds_all_bots),Xss_C,Fss_C(Xss_ia));
    
    est_mu = abs(pred_h);
    est_s2 = abs(pred_Var);
    %     est_hyp = hyp_gpml;
    %Change this maybe to make beta large and alpha*est_mu?
    %phi_func = kappa*est_mu + beta*est_s2;
    
    % compute MI lower bound as phi for optimizing sample selection
    phi_func = zeros(length(Fss),1);
    for iij = 1:num_bot
        phi_func(k == iij) = kappa*est_mu(k == iij) + beta*(sqrt(est_s2(k == iij) + gamma_gp(iij)) - sqrt(gamma_gp(iij)));
    end
    
    % create train test split
    idx_train = unique([bots.Nm_ind]);
    idx_test = setdiff(1:length(Fss),idx_train);
    
    % collect metrics for plotting
    rms_stack(it) = sqrt(sum( (pred_h(idx_test) - Fss(idx_test)).^2 )/(length(Fss(idx_test)))); %mean(pred_rms);
    avg_rms_stack(it) = mean(rms_stack, "omitnan");
%     if it ~= 1
%         rms_stack(it) = (rms_stack(it-1)*(it-1) + rms_stack(it))/it;
%     end
    var_stack(it) = mean(pred_Var);
    [max_mis(it), ~] = max(abs(Fss(idx_test)-est_mu(idx_test)));
    avg_max_mis(it) = mean(max_mis, "omitnan");

    % calculate Vornoi centroid (debug only, not used for controller though)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    g_new = g;
    
    m = zeros(g_num,1);
    accumM = accumarray (k, phi_func);   % this computes \sigma_{V_i} \phi(q) \dq for all voronoi cell V_i  % ones(s_num,1)
    m(1:length(accumM)) = accumM;  % \sigma_{V_i} \phi(q) \dq for each V   (g_num x 1)
    
    sumx = zeros(g_num,1);
    sumy = zeros(g_num,1);
    accumX = accumarray ( k, s(1,:)'.*phi_func );     %   \sigma_{V_i} \q_x \phi(q) \dq   ucb
    accumY = accumarray ( k, s(2,:)'.*phi_func );   %  \sigma_{V_i} \q_y \phi(q) \dq     ucb
    sumx(1:length(accumX)) = accumX;
    sumy(1:length(accumY)) = accumY;  % same as above
    g_new(1,m~=0) = sumx(m~=0) ./ m(m~=0);   % get x coordinate for the new centroid
    g_new(2,m~=0) = sumy(m~=0) ./ m(m~=0);   % get y coordinate for the new centroid
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % get start locs and idxs
    g_start = g;
    proj_g_idx_start = get_idx(g_start', Xss);
    % collect shortest paths for each robot
    shortest_paths = [];
    % iterate over robots and calculate shortest paths to each loc in env
    for ijkl = 1:num_bot
        % kth max value
        curr_k = 1;
        % locate most informative loc that is reachable under shortest path
        while true
            start_idx = Xss(proj_g_idx_start(ijkl),1:2);
            mod_Xss = Xss(k==ijkl, :);
            %mod_Xss_no_time = mod_Xss(:, 1:2);
            dists = pdist2(mod_Xss(:, 1:2), start_idx, 'cityblock');
            mod_phi_func = phi_func(k==ijkl);
            mod_phi_func(dists > max_path_length) = 0;
            if useRatioPath == true
                dist_denom = (1/(max_path_length)).*(dists - max_path_length).^2 + 1;
                mod_phi_func = mod_phi_func ./ dist_denom;
                %mod_phi_func = phi_func(k==ijkl)./dists;
                %mod_phi_func = mod_phi_func ./ dists;
            end
            mod_phi_func(isinf(mod_phi_func)) = 0;
            [M,I] = maxk(mod_phi_func, curr_k);
            g_new(:,ijkl) = mod_Xss(I(end), :)';
            g_new = g+kp*(g_new-g);
            proj_g_idx = get_idx(g_new', Xss);
%             start_idx = Xss(proj_g_idx_start(ijkl),1:2);
            start = 45*start_idx(1) + start_idx(2) + 1;
            goal_idx = Xss(proj_g_idx(ijkl),1:2);
            goal = 45*goal_idx(1)  + goal_idx(2) + 1;
            try
                shortestP = shortestpath(Agraph, start, goal);
            catch
                warning('Problem using shortest path function.');
            end
            if length(shortestP) <= max_path_length
                shortest_paths = [shortest_paths length(shortestP)];
                if length(shortestP) < max_path_length
                    disp('hey a shorter path')

                end
                break;
            else
                curr_k = curr_k + 1;
            end
        end
    end
    
    % Calculate actual centroid location (not used in algo, only for debug)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    g_actual = g;
    
    m = zeros(g_num,1);
    accumM = accumarray (k, Fss);   % this computes \sigma_{V_i} \phi(q) \dq for all voronoi cell V_i  % ones(s_num,1)
    m(1:length(accumM)) = accumM;  % \sigma_{V_i} \phi(q) \dq for each V   (g_num x 1)
    
    accumX = accumarray ( k, s(1,:)'.*Fss );     %   \sigma_{V_i} \q_x \phi(q) \dq   actual density function
    accumY = accumarray ( k, s(2,:)'.*Fss );   %  \sigma_{V_i} \q_y \phi(q) \dq     actual density function
    sumx(1:length(accumX)) = accumX;
    sumy(1:length(accumY)) = accumY;
    g_actual(1,m~=0) = sumx(m~=0) ./ m(m~=0);   % get x coordinate for the actual centroid
    g_actual(2,m~=0) = sumy(m~=0) ./ m(m~=0);   % get y coordinate for the actual centroid
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % plot GT data in curr timestep vs estimate from MGP and save figs
    figure(51);
    plot_surf2( ksx_g,ksy_g,reshape(Fss_big(945*(it-1)+1:945*it),[size(ksx_g,1),size(ksx_g,2)])', map_x, map_y, map_z,25, 5      );
    hold on;
%     lineStyles = linspecer(10);
    colormap(linspecer);
    saveas(gcf, string(it) + 'gt.png');

    figure(52);
    plot_surf2( ksx_g,ksy_g,reshape(est_mu(size(temp_Fss,1)+1:end),[size(ksx_g,1),size(ksx_g,2)])', map_x, map_y, map_z,25, 5      );
    hold on;
%     lineStyles = linspecer(10);
    colormap(linspecer);
    saveas(gcf, string(it) + 'est.png');
    
    
    if ismember(it,[1 11])
        pause(0.1)
        figure(50);
    else
        figure(50) % 20
    end
    
    
    %         hold on;
    %plot_surf2( ksx_g,ksy_g,reshape(est_mu,[size(ksx_g,1),size(ksx_g,2)]), map_x, map_y, map_z,25, 5      );

    % plot current MGP estimate with Voronoi partitions and robot locs
    plot_surf2( ksx_g,ksy_g,reshape(est_mu(size(temp_Fss,1)+1:end),[size(ksx_g,1),size(ksx_g,2)]), map_x, map_y, map_z,25, 5      );
    
    hold on;
    lineStyles = linspecer(10);
    colormap(linspecer);
    hold on;
    [edge_x, edge_y] = voronoi(g(1,:),g(2,:),t); %'r--'
    hold on;
    %     plot(g(1,:),g(2,:), 'b+');
    
    %     if it~=1
    for idx_plot = 1:g_num
        
        plot(bots(idx_plot).Xs(num_init_sam:end,1),bots(idx_plot).Xs(num_init_sam:end,2),'LineWidth',5,'Color',lineStyles(1,:));
        hold on;
        plot(bots(idx_plot).Xs(end,1),bots(idx_plot).Xs(end,2),'o','MarkerSize',20,'LineWidth',5,'Color',lineStyles(2,:))
        hold on;
    end
    %     end
    
    
    hold on;
    plot(g_actual(1,:),g_actual(2,:), '*','MarkerSize',20,'LineWidth',5,'Color',lineStyles(9,:));
    
    hold on;
    plot(edge_x,edge_y,'--','Color',lineStyles(9,:),'LineWidth',5);
    
    hold on;
    
    text(g(1,:)'+2,g(2,:)',int2str((1:g_num)'),'FontSize',25);
    %     title_string = sprintf ( 'Weighted Voronoi' );
    %     title ( title_string );
%     axis equal
    axis ([  0, map_x, 0, map_y ])
    %     axis square
    drawnow
    xlabel('X','FontSize',25);
    ylabel('Y','FontSize',25);
    %     zlabel('Temperature (degrees Celsius)','FontSize',20);
    %     caxis([15,29]);
    set(gca,'LineWidth',5,'fontsize',25,'fontname','Times','FontWeight','Normal');
    %     grid on;
    
    hold off;
    
    if ismember(it,[1 11])
        pause(0.1)
    end

    % Compute sensing cost function (not used, only for debug now)
    
    cf(it) = 0;
    for i = 1:s_num
        cf(it) = cf(it)+((s(1,i)-g(1,k(i))).*Fss(i))^2+((s(2,i)-g(2,k(i))).*Fss(i)) ^2;  % get summation of distance to goal
    end
    
    %  Display the energy.
    %

    % Plot RMSE error
    figure(21);
    subplot ( 1, 2, 1 )

    %         plot ( step, e )
    plot(step, rms_stack, 'r-s');
    title ( 'RMS Error' )
    xlabel ( 'Step' )
    ylabel ( 'RMSE' )
    %         ylabel ( 'Weightings' )
    grid on
    axis equal
    xlim([0 it_num]);
    %         ylim([-0.08 0.08])
    axis square
    drawnow

    % Plot max mismatch and avg max mismatch (not used, just for debug)

    figure(30);
    subplot ( 1, 2, 1 )
    plot(step, max_mis);
    title('Max Mismatch')
    xlabel('Step')
    ylabel('Max Mismatch')
    grid on
    axis equal
    xlim([0 it_num]);
    axis square
    drawnow

    figure(30);
    subplot ( 1, 2, 2 )
    plot(step, avg_max_mis);
    title('Average Max Mismatch Over Timesteps')
    xlabel('Step')
    ylabel('Average Max Mismatch')
    grid on
    axis equal
    xlim([0 it_num]);
    axis square
    drawnow
% 
%     
%     %
%     %  Compute the generator motion.
%     
%     %
%     %  Display the generator motion.
%     %

    % Plot Average RMSE error
    figure(21);
    subplot ( 1, 2, 2 )
%     
%     
%     %     figure(22);
    plot ( step, avg_rms_stack, 'm-' )
    title ( 'Average RMSE Over Timesteps' )
    xlabel ( 'Step' )
    ylabel ( 'Average RMSE' )
    grid on;
    axis equal
    xlim([0 it_num]);
    
    axis square
    drawnow
%     
%     pause(1);
    
    %
    %  Update the generators.

%     for ijkl = 1:num_bot
%         [M,I] = maxk(phi_func(k==ijkl), 1);
%         g_new(:,ijkl) = Xss(I, :)';
%     end
    g_start = g;
    proj_g_idx_start = get_idx(g_start', Xss);
    g = g+kp*(g_new-g); %g_new; %
    %g = g_new;
    
    %     proj_g_idx = get_idx(g', Xss);
    
    proj_g_idx = get_idx(g', Xss);  % sample the actually visited point instead of the centroid
    stop_count_ipp = 0;

    % Construct mutual info approx using submodular function optimization
    mask = (Xss_C(:,3) == it);
    n = sum(mask);
    V = 1:n;
    F = sfo_fn_mi(pred_post, V);

    % search set of paths to find best informative path
    for ijk = 1:g_num
        start_idx = Xss(proj_g_idx_start(ijk),1:2);
        start = 45*start_idx(1) + start_idx(2) + 1;
        goal_idx = Xss(proj_g_idx(ijk),1:2);
        goal = 45*goal_idx(1)  + goal_idx(2) + 1;
        Ps = allpaths(Agraph, start, goal, "MaxPathLength",max_path_length);
        if useApproxMI == true
            % Use secretary algo to find approx good waypoint/sample
            a2 = 0;
            k = 1;
            l = 1;
            n = length(Ps);
            for t = (floor((l - 1)*n/k))+1:(floor(l*n/k))
                P = cell2mat(Ps(t));
                mInfo_2 = F(P);
                if (l - 1)*n/k < t && t < (l - 1)*n/k + n/(k*exp(1))
                    if mInfo_2 > a2
                        a2 = mInfo_2;
                    end
                else
                    if (mInfo_2 > a2 || t == floor(l*n/k))
                        path_to_take = P;
                        break;
                    end
                end
            end
        else
            % search for maximally informative waypoint/sample
            max_mInfo = 0;
            for paths = 1:length(Ps)
                P = cell2mat(Ps(paths));
                mInfo = F(P);
                if mInfo > max_mInfo
                    path_to_take = P;
                    max_mInfo = mInfo;
                end
            end
        end
        P = path_to_take;
        % Collect samples from chosen path and add to robot data
        for p = 2:length(P)-1
            curr_loc = P(p);
            y_idx = mod(curr_loc, 45) - 1;
            if y_idx == -1
                y_idx = 44;
            end
            x_idx = (curr_loc - y_idx - 1)/45;
            samp_loc = [x_idx y_idx it];
            temp_proj_g_idx = find(ismember(Xss, samp_loc,'rows'));
            try
                temp_proj_g_idx = temp_proj_g_idx(end);
            catch
                warning('Problem using function.  temp_proj_g_idx empty.');
            end
            if bots(ijk).Nm_ind(end) ~= temp_proj_g_idx
                bots(ijk).Nm_ind(end+1) = temp_proj_g_idx;
                bots(ijk).Xs(end+1,:) = Xss(temp_proj_g_idx,:);
            else
                stop_count_ipp = stop_count_ipp + 1;
            end
            bots(ijk).Nm_ind = bots(ijk).Nm_ind(:)';
        end
    end
    
    stop_count = 0;
    for ijk = 1:g_num
        gamma_gp(ijk) = gamma_gp(ijk) + est_s2(proj_g_idx(ijk),:);
        if bots(ijk).Nm_ind(end) ~= proj_g_idx(ijk)
            bots(ijk).Nm_ind(end+1) = proj_g_idx(ijk);
            bots(ijk).Xs(end+1,:) = Xss(proj_g_idx(ijk),:);
        else
            stop_count = stop_count + 1;
        end
        bots(ijk).Nm_ind = bots(ijk).Nm_ind(:)';
    end
    
    % after consensus loop, updated variable:  1) bots.neighbor, 2)
    % bots.Nm_ind, 3) new local model, 4) packets
    %     bots = updateBotModel(bots);
    
    
    if stop_count == num_bot
        stop_flag = 1;
    end

    if ~isempty(lastwarn(''))
        sweep
        failures = [failures sweep]
        %error('sweep fail')
    end
    
    
end

date_time = datetime('now');
dat_name = strcat('sim1_gmm_',num2str(date_time.Month), '_', num2str(date_time.Day), '_', ...
    num2str(date_time.Year), '_', num2str(date_time.Hour), '_', ...
    num2str(date_time.Minute), '_', num2str(round(date_time.Second)), '.mat');

if save_flag
    save(dat_name);  %save('iros_sim2_decgp_temp_t.mat');
end

%end %for loop of sweep
end


function idx = get_idx(Xtest, Xs)
Distance = pdist2(Xtest, Xs,'euclidean');
[~, idx] = min(Distance,[],2);
end

% Transmit bot n's packet to any neighbors for whom packetLost returns false
function bots = transmitPacket(bots, n)
num_bot = numel(bots);
for j=1:num_bot
    %if ~packetLost(norm(bots(n).state.p - bots(j).state.p))
    if ismember(j, bots(n).neighbor)
        bots(j).packets(n) = bots(n).packets(n);
    end
    %end
end
end

% update cycle

function bots = updateBotModel(bots)
global num_bot num_gau Xss Fss
packets = struct('alpha_K',[],'belta_K',[],'gamma_K',[]);

for ijk = 1:length(bots)
    model = struct();
    
    model.Sigma = bots(ijk).gamma_K./bots(ijk).alpha_K;
    
    kss = zeros(1,1,num_gau);
    for ijj = 1:num_gau
        kss(:,:,ijj) =  model.Sigma(ijj);
    end
    model.Sigma = kss;
    
    model.mu = bots(ijk).belta_K./(bots(ijk).alpha_K);
    model.w = norm_prob(bots(ijk).alpha_K);
    
    
    [~, model, ~, ~, ~] = mixGaussEm_rss(Fss(bots(ijk).Nm_ind)', model); % initialize with converged previous mu, Sigma, alpha, and new Nm_ind
    Nm = length(bots(ijk).Nm_ind);
    bots(ijk).mu_K = model.mu;
    bots(ijk).Sigma_K = model.Sigma;
    bots(ijk).self_alpha = model.w;
    %             bots(ijk).alpha_K = model.w;
    
    [~, alpha_mnk] = mixGaussPred_rss(Fss(bots(ijk).Nm_ind)', model); % get alpha_mnk:   Nm x num_gau
    self_alpha = sum(alpha_mnk,1); %./Nm;  % 1 x num_gau    Nm*alpha_mk
    y_mn = Fss(bots(ijk).Nm_ind);    %   Nm x 1
    bots(ijk).belta_K = sum(alpha_mnk.*y_mn,1);  %  1 x num_gau  belta_mk
    % mu_K = self_belta./(Nm*self_alpha);  % 1 x num_gau   mu_mk
    bots(ijk).gamma_K = sum(((repmat(y_mn,[1,num_gau])-model.mu).^2).*alpha_mnk, 1);  %  1 x num_gau
    bots(ijk).alpha_K = self_alpha;
    
    for ijk_rec = 1:num_bot
        bots(ijk).packets(ijk_rec) = packets;   % initialize packets struct for every robots
    end
    bots(ijk).neighbor = setdiff(1:num_bot, ijk);    % find(adj_A(ijk,:)>0);  % get neighbor id, intialize from a fully connected graph
    
end

for ijk = 1:num_bot     % reset packets to the bot themselves
    packets.alpha_K = bots(ijk).alpha_K;
    packets.belta_K = bots(ijk).belta_K;
    packets.gamma_K = bots(ijk).gamma_K;
    bots(ijk).packets(ijk) = packets;
end


end



function bots = updateBotComputations(bots, n)
global Fss eta
num_gau = numel(bots(n).alpha_K);
Nm = length(bots(n).Nm_ind);        %  initialization

% resort bot mu and ind
% [~, resort_ind] = sort(bots(n).mu_K, 'ascend'); % in case some components are with different orders during each node local computation
% bots(n).mu_K = bots(n).mu_K(resort_ind);
% bots(n).Sigma_K = bots(n).Sigma_K(:,:,resort_ind);
% bots(n).alpha_K = bots(n).alpha_K(resort_ind);

model = struct;
model.mu = bots(n).mu_K;
model.Sigma = bots(n).Sigma_K;
model.w = norm_prob(bots(n).alpha_K); %bots(n).alpha_K;

% mu_k = ;
% sigma_k = zeros(1,num_gau);
% gamma_k = zeros(1,num_gau);


%      [label, model, llh, break_flag] = mixGaussEm_rss(Fss(bots(n).Nm_ind)', num_gau);
[~, alpha_mnk] = mixGaussPred_gmm(Fss(bots(n).Nm_ind)', model); % get alpha_mnk:   Nm x num_gau
self_alpha = sum(alpha_mnk,1); %./Nm;  % 1 x num_gau    Nm*alpha_mk
y_mn = Fss(bots(n).Nm_ind);    %   Nm x 1
self_belta = sum(alpha_mnk.*y_mn,1);  %  1 x num_gau  belta_mk
% mu_K = self_belta./(Nm*self_alpha);  % 1 x num_gau   mu_mk
self_gamma = sum(((repmat(y_mn,[1,num_gau])-model.mu).^2).*alpha_mnk, 1);  %  1 x num_gau
% self_Sigma = reshape(self_gamma./(Nm*self_alpha),[1,1,num_gau]);  % 1 x num_gau   gamma_mk

bots(n).self_alpha = self_alpha;
bots(n).self_belta = self_belta;
bots(n).self_gamma = self_gamma;

%% after compute local summary stats, we update estimate of global stats using packets
% without considering age of the packets

num_neighbor = length(bots(n).neighbor);

%% start consensus based dynamic estimation process
stack_alpha_neighbor = reshape([bots(n).packets(bots(n).neighbor).alpha_K].',[num_gau, num_neighbor]).';
stack_belta_neighbor = reshape([bots(n).packets(bots(n).neighbor).belta_K].',[num_gau, num_neighbor]).';
stack_gamma_neighbor = reshape([bots(n).packets(bots(n).neighbor).gamma_K].',[num_gau, num_neighbor]).';

bots(n).dot_alpha_K = sum(stack_alpha_neighbor - bots(n).alpha_K,1) + bots(n).self_alpha - bots(n).alpha_K; %  note the difference self_alpha should be Nm*alpha or just alpha ?
bots(n).dot_belta_K = sum(stack_belta_neighbor - bots(n).belta_K,1) + bots(n).self_belta - bots(n).belta_K;
bots(n).dot_gamma_K = sum(stack_gamma_neighbor - bots(n).gamma_K,1) + bots(n).self_gamma - bots(n).gamma_K;

bots(n).alpha_K = bots(n).alpha_K + eta*bots(n).dot_alpha_K;
bots(n).belta_K = bots(n).belta_K + eta*bots(n).dot_belta_K;
bots(n).gamma_K = bots(n).gamma_K + eta*bots(n).dot_gamma_K;


bots(n).Sigma_K = bots(n).gamma_K./bots(n).alpha_K;
bots(n).mu_K = bots(n).belta_K./(bots(n).alpha_K);

kss = zeros(1,1,num_gau);
for ijj = 1:num_gau
    
    if bots(n).Sigma_K(ijj)<10^-5
        disp('Sigma too low')
        pause;
    end
    
    kss(:,:,ijj) = bots(n).Sigma_K(ijj);
end
bots(n).Sigma_K = kss;




%% end of estimation and parameter updates


bots(n).packets(n).alpha_K = bots(n).alpha_K;
bots(n).packets(n).belta_K = bots(n).belta_K;
bots(n).packets(n).gamma_K = bots(n).gamma_K;

end



function y = loggausspdf(X, mu, Sigma)
d = size(X,1);   %  X:   d x Nm
X = bsxfun(@minus,X,mu);
[U,p]= chol(Sigma);
if p ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;
end

function y = norm_prob(X)
% X:  n x d where d is the num_gau
y = X./sum(X,2);

end

% Create lattice Adjacency Matrix for creating graph for IPP
function A = latticeAdjacencyMatrix(N,M)
  % N rows, M columns, denoting the size of the rectangular lattice
  % A - N*M by N*M square adjacency matrix
  % Connect nodes (i,j) to (i+1,j)
  [i,j] = ndgrid(1:N-1,1:M);
  ind1 = sub2ind([N,M],i,j);
  ind2 = sub2ind([N,M],i+1,j);
  
  % Connect nodes (i,j) to (i,j+1)
  [i,j] = ndgrid(1:N,1:M-1);
  ind3 = sub2ind([N,M],i,j);
  ind4 = sub2ind([N,M],i,j+1);
  
  % build the global adjacency matrix
  totalnodes = N*(M-1) + (N-1)*M;
  A = sparse([ind1(:);ind3(:)],[ind2(:);ind4(:)],ones(totalnodes,1),N*M,N*M);
  
  % symmetrize, since the above computations only followed the edges in one direction.
  % that is to say, if a talks to b, then b also talks to a.
  A = A + A';
end