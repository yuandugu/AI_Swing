%% start and prepare
clear all
clear global
T=120;% T>6
step=0.01;
l=21;
nn=2;
N=2;
m=2;
jay = sqrt(-1);
pst_var % set up global variables
svc_dc=[];tcsc_dc=[];dcr_dc=[];dci_dc=[];
% load input data from m.file
disp('non-linear simulation')
% input data file
dfile='datane1.m';
lfile =length(dfile);
% strip off .m and convert to lower case
dfile = lower(dfile(1:lfile-2));
eval(dfile);
sw_con(end,1)=T;
% check for valid dynamic data file
basdat = {'100';'50'};
sys_freq = str2double(basdat{2});
basrad = 2*pi*sys_freq; % default system frequency is 50 Hz
basmva = str2double(basdat{1});
syn_ref = 0 ;     % synchronous reference frame
ibus_con = []; % ignore infinite buses in transient simulation

% solve for loadflow - loadflow parameter
dcsp_con=[];
n_conv = 0;
n_dcl = 0;
ndcr_ud=0;
ndci_ud=0;
tol = 1e-8;   % tolerance for convergence
iter_max = 30; % maximum number of iterations
acc = 1.0;   % acceleration factor
[bus_sol,line,line_flw] = ...
    loadflow(bus,line,tol,iter_max,acc,'n',2);
bus = bus_sol;  % solved loadflow solution needed for
% initialization
save sim_fle bus line
% construct simulation switching sequence as defined in sw_con
tswitch(1) = sw_con(1,1);
k = 1;kdc=1;
n_switch = length(sw_con(:,1));
k_inc = zeros(n_switch-1,1);k_incdc=k_inc;
t_switch = zeros(n_switch,1);
h=t_switch;h_dc=h;
for sw_count = 1:n_switch-1
   h(sw_count) = sw_con(sw_count,7);%specified time step
   if h(sw_count)==0, h(sw_count) = 0.01;end % default time step
   k_inc(sw_count) = fix((sw_con(sw_count+1,1)-sw_con(sw_count,1))/h(sw_count));%nearest lower integer
   if k_inc(sw_count)==0;k_inc(sw_count)=1;end% minimum 1
   h(sw_count) = (sw_con(sw_count+1,1)-sw_con(sw_count,1))/k_inc(sw_count);%step length
   h_dc(sw_count) = h(sw_count)/10;
   k_incdc(sw_count) = 10*k_inc(sw_count);
   t_switch(sw_count+1) =t_switch(sw_count) +  k_inc(sw_count)*h(sw_count);
   t(k:k-1+k_inc(sw_count)) = t_switch(sw_count):h(sw_count):t_switch(sw_count+1)-h(sw_count);
   t_dc(kdc:kdc-1+k_incdc(sw_count)) = t_switch(sw_count):h_dc(sw_count):t_switch(sw_count+1)-h_dc(sw_count);
   k=k+k_inc(sw_count);kdc=kdc+k_incdc(sw_count);
end
t_dc(kdc)=t_dc(kdc-1)+h_dc(sw_count);
for kk=1:10;kdc=kdc+1;t_dc(kdc)=t_dc(kdc-1)+h_dc(sw_count);end

k = sum(k_inc)+1; % k is the total number of time steps in the simulation

t(k) = sw_con(n_switch,1);
%% loop
% perturbation prepare
if l > 1
    x = -pi:2*pi/(l-1):pi;
    y = -1/5/pi:2/5/pi/(l-1):1/5/pi;
    [X,Y] = meshgrid(x,y);
    x=reshape(X,[1,l^2]);
    y=reshape(Y,[1,l^2]);
elseif l == 1
    x=0;
    y=0;
end
i=0;
index=fix(l^2/nn)*ones(1,nn);
for i=1:mod(l^2,nn)
    index(1,i)=index(1,i)+1;
end
index_start=ones(1,nn);
index_end=zeros(1,nn);
for i=1:nn
    if i > 1;index_start(1,i)=index_start(1,i)+sum(index(1,1:i-1));end
    index_end(1,i)=sum(index(1,1:i));
end
% i=0;
% save data prepare
len=length(t);
bus_theta=zeros(index(1,m),39,len);
bus_voltage=zeros(index(1,m),39,len);
gen_theta=zeros(index(1,m),10,len);
gen_speed=zeros(index(1,m),10,len);
ind_slip=zeros(index(1,m),29,len);
% start
for i=1:index(1,m)
    clear pst_var
    pst_var % set up global variables
    svc_dc=[];tcsc_dc=[];dcr_dc=[];dci_dc=[];
    syn_ref = 0 ;     % synchronous reference frame
    ibus_con = []; % ignore infinite buses in transient simulation
    n_conv = 0;
    n_dcl = 0;
    ndcr_ud=0;
    ndci_ud=0;
    bus = bus_sol;
    %set indexes
    % note: dc index set in dc load flow
    mac_indx;
    exc_indx;
    tg_indx;
    dpwf_indx;
    pss_indx;
    svc_indx(svc_dc);
    tcsc_indx(tcsc_dc);
    lm_indx;
    rlm_indx;
    n_mot = size(ind_con,1);
    n_ig = size(igen_con,1);
    if isempty(n_mot); n_mot = 0;end
    if isempty(n_ig); n_ig = 0; end
    ntot = n_mac+n_mot+n_ig;
    ngm = n_mac + n_mot;
    n_pm = n_mac;
    disp(' ')
    disp('Performing simulation.')
    %
    % construct simulation switching sequence as defined in sw_con
    tswitch(1) = sw_con(1,1);
    k = 1;kdc=1;
    n_switch = length(sw_con(:,1));
    k_inc = zeros(n_switch-1,1);k_incdc=k_inc;
    t_switch = zeros(n_switch,1);
    h=t_switch;h_dc=h;
    for sw_count = 1:n_switch-1
       h(sw_count) = sw_con(sw_count,7);%specified time step
       if h(sw_count)==0, h(sw_count) = 0.01;end % default time step
       k_inc(sw_count) = fix((sw_con(sw_count+1,1)-sw_con(sw_count,1))/h(sw_count));%nearest lower integer
       if k_inc(sw_count)==0;k_inc(sw_count)=1;end% minimum 1
       h(sw_count) = (sw_con(sw_count+1,1)-sw_con(sw_count,1))/k_inc(sw_count);%step length
       h_dc(sw_count) = h(sw_count)/10;
       k_incdc(sw_count) = 10*k_inc(sw_count);
       t_switch(sw_count+1) =t_switch(sw_count) +  k_inc(sw_count)*h(sw_count);
       t(k:k-1+k_inc(sw_count)) = t_switch(sw_count):h(sw_count):t_switch(sw_count+1)-h(sw_count);
       t_dc(kdc:kdc-1+k_incdc(sw_count)) = t_switch(sw_count):h_dc(sw_count):t_switch(sw_count+1)-h_dc(sw_count);
       k=k+k_inc(sw_count);kdc=kdc+k_incdc(sw_count);
    end
    t_dc(kdc)=t_dc(kdc-1)+h_dc(sw_count);
    for kk=1:10;kdc=kdc+1;t_dc(kdc)=t_dc(kdc-1)+h_dc(sw_count);end

    k = sum(k_inc)+1; % k is the total number of time steps in the simulation

    t(k) = sw_con(n_switch,1);
    [n dummy]=size(mac_con) ;
    n_bus = length(bus(:,1));

    % create zero matrices for variables to make algorithm more efficient?
    z = zeros(n,k);
    z1 = zeros(1,k);
    zm = zeros(1,k);if n_mot>1;zm = zeros(n_mot,k);end
    zig = zeros(1,k);if n_ig>1;zig = zeros(n_ig,k);end
    zdc = zeros(2,kdc);if n_conv>2; zdc = zeros(n_conv,kdc);end
    zdcl = zeros(1,kdc);if n_dcl>1;zdcl=zeros(n_dcl,kdc);end
    % set dc parameters  
    Vdc = zeros(n_conv,kdc);
    i_dc = zdc;  
    P_dc = z; cur_ord = z;
    alpha = zdcl; 
    gamma = zdcl;  
    dc_sig = zeros(n_conv,k);dcr_dsig = zeros(n_dcl,k);dci_dsig=zeros(n_dcl,k);
    i_dcr = zdcl; i_dci = zdcl; v_dcc = zdcl;
    di_dcr = zdcl; di_dci = zdcl; dv_dcc = zdcl;
    v_conr = zdcl; v_coni = zdcl; dv_conr = zdcl; dv_coni = zdcl;
    
    v_p = z1;
    theta = zeros(n_bus+1,k);bus_v = zeros(n_bus+1,k);
    mac_ang = z; mac_spd = z; dmac_ang= z; dmac_spd = z;
    pmech = z; pelect = z; mac_ref = z1;  sys_ref = z1; 
    edprime = z; eqprime = z; dedprime = z; deqprime = z;
    psikd = z; psikq = z; dpsikd = z; dpsikq = z;
    pm_sig = z;
    z_tg = zeros(1,k);if n_tg+n_tgh~=0;z_tg = zeros(n_tg+n_tgh,k);end
    tg1 = z_tg; tg2 = z_tg; tg3 = z_tg; tg4 = z_tg; tg5 = z_tg;
    dtg1 = z_tg; dtg2 = z_tg; dtg3 = z_tg;dtg4 = z_tg; dtg5 = z_tg;
    tg_sig = z_tg;
    z_pss = zeros(1,k);if n_pss~=0;z_pss = zeros(n_pss,k);end
    pss1 = z_pss; pss2 = z_pss; pss3 = z_pss;
    dpss1 = z_pss; dpss2 = z_pss; dpss3 = z_pss;
    z_dpw = zeros(1,k);if n_dpw~=0; z_dpw = zeros(n_dpw,k);end
    sdpw1 = z_dpw; sdpw2 = z_dpw; sdpw3 = z_dpw; sdpw4 = z_dpw; sdpw5 = z_dpw; sdpw6 = z_dpw; dpw_out = z_dpw;
    dsdpw1 = z_dpw; dsdpw2 = z_dpw; dsdpw3 = z_dpw; dsdpw4 = z_dpw; dsdpw5 = z_dpw; dsdpw6 = z_dpw;
    curd = z; curq = z; curdg = z; curqg = z; fldcur = z;
    ed = z; eq = z; eterm = z; qelect = z;
    vex = z; cur_re = z; cur_im = z; psi_re = z; psi_im = z;
    ze = zeros(1,k);if n_exc~=0; ze = zeros(n_exc,k);end
    V_B = ze;exc_sig = ze;
    V_TR = ze; V_R = ze; V_A = ze; V_As = ze; Efd = ze; R_f = ze;
    dV_TR = ze; dV_R = ze; dV_As = ze; dEfd = ze; dR_f = ze;
    pss_out = ze;
    vdp = zm; vqp = zm; slip = zm; 
    dvdp = zm; dvqp = zm; dslip = zm;
    s_mot = zm; p_mot = zm; q_mot = zm;
    vdpig = zig; vqpig = zig; slig = zig;
    dvdpig = zig; dvqpig = zig; dslig = zig;
    s_igen = zig; pig = zig; qig = zig; tmig = zig;
    if n_svc~=0
       B_cv = zeros(n_svc,k); dB_cv = zeros(n_svc,k);svc_sig = zeros(n_svc,k);svc_dsig=zeros(n_svc,k);
       B_con = zeros(n_svc,k); dB_con=zeros(n_svc,k);
       if n_dcud~=0
          d_sig = zeros(n_dcud,k);
          for j = 1:n_dcud
             sv = get(svc_dc{j,1});
             if j==1
                xsvc_dc =zeros(sv.NumStates,k); 
                dxsvc_dc = zeros(sv.NumStates,k);
             else
                xsvc_dc = [xsvc_dc;zeros(sv.NumStates,k)];
                dxsvc_dc = [dxsvc_dc;zeros(sv.NumStates,k)];
             end
          end
       else
          xsvc_dc = zeros(1,k);
          dxsvc_dc = zeros(1,k);
       end
    else
       B_cv = zeros(1,k);dB_cv = zeros(1,k); svc_sig = zeros(1,k);svc_dsig = zeros(1,k);
       B_con = zeros(1,k);dB_con=zeros(1,k);
       xsvc_dc = zeros(1,k);dxsvc_dc = zeros(1,k);
       d_sig = zeros(1,k);
    end
    if n_tcsc~=0
       B_tcsc = zeros(n_tcsc,k); dB_tcsc = zeros(n_tcsc,k);tcsc_sig = zeros(n_tcsc,k);tcsc_dsig=zeros(n_tcsc,k);
       if n_tcscud~=0
          td_sig = zeros(n_tcscud,k);%input to tcsc damping control
          for j = 1:n_tcscud
             sv = get(tcsc_dc{j,1});% damping control state space object
             if j==1
                xtcsc_dc =zeros(sv.NumStates,k); % tcsc damping control states
                dxtcsc_dc = zeros(sv.NumStates,k);% tcsc dc rates of chage of states
             else
                xtcsc_dc = [xtcsc_dc;zeros(sv.NumStates,k)];% in order of damping controls
                dxtcsc_dc = [dxtcsc_dc;zeros(sv.NumStates,k)];
             end
          end
       else
          xtcsc_dc = zeros(1,k);
          dxtcsc_dc = zeros(1,k);
       end
    else
       B_tcsc = zeros(1,k);dB_tcsc = zeros(1,k); tcsc_sig = zeros(1,k);tcsc_dsig = zeros(1,k);
       xtcsc_dc = zeros(1,k);dxtcsc_dc = zeros(1,k);
       td_sig = zeros(1,k);
    end

    if n_lmod ~= 0
       lmod_st = zeros(n_lmod,k); dlmod_st = lmod_st; lmod_sig = lmod_st;
    else
       lmod_st = zeros(1,k); dlmod_st = lmod_st; lmod_sig = lmod_st;
    end
    if n_rlmod ~= 0
       rlmod_st = zeros(n_rlmod,k); drlmod_st = rlmod_st; rlmod_sig = rlmod_st;
    else
       rlmod_st = zeros(1,k); drlmod_st = rlmod_st; rlmod_sig = rlmod_st;
    end

    sys_freq = ones(1,k);
    disp('constructing reduced y matrices')
    % step 1: construct reduced Y matrices 

    disp('initializing motor,induction generator, svc and dc control models')       
    bus = mac_ind(0,1,bus,0);% initialize induction motor
    bus = mac_igen(0,1,bus,0); % initialize induction generator
    bus = svc(0,1,bus,0);%initialize svc
    f = dc_cont(0,1,1,bus,0);% initialize dc controls
    % this has to be done before red_ybus is used since the motor and svc 
    % initialization alters the bus matrix and dc parameters are required

    y_switch % calculates the reduced y matrices for the different switching conditions
    disp('initializing other models')
    % step 2: initialization
    theta(1:n_bus,1) = bus(:,3)*pi/180;
    bus_v(1:n_bus,1) = bus(:,2).*exp(jay*theta(1:n_bus,1));

    flag = 0;
    bus_int = bus_intprf;% pre-fault system
    disp('generators')
    mac_sub(0,1,bus,flag);
    mac_tra(0,1,bus,flag);
    mac_em(0,1,bus,flag);
    disp('generator controls')
    dpwf(0,1,bus,flag);
    pss(0,1,bus,flag);
    smpexc(0,1,bus,flag);
    smppi(0,1,bus,flag);
    exc_st3(0,1,bus,flag);
    exc_dc12(0,1,bus,flag);
    tg(0,1,bus,flag); 
    tg_hydro(0,1,bus,flag);

    nload = 0;
    
    H_sum = sum(mac_con(:,16)./mac_pot(:,1));
    tic % set timer
    % step 3: perform a predictor-corrector integration
    kt = 0;
    ks = 1;
    k_tot = sum(k_inc);
    lswitch = length(k_inc);
    ktmax = k_tot-k_inc(lswitch);
    bus_sim = bus;
    plot_now = 0;
    mac_ang(N,1)=mac_ang(N,1)+x(1,index_start(1,m)+i-1);
    mac_spd(N,1)=mac_spd(N,1)+y(1,index_start(1,m)+i-1);
    % slip(2,1)=slip(2,1)+0.00;
    % theta(2,1)=theta(2,1)+0;
    % bus_v(2,1)=abs(bus_v(2,1))*exp(1i*(angle(bus_v(2,1))+2));
    while (kt<=ktmax)
        k_start = kt+1;
        if kt==ktmax
            k_end = kt + k_inc(ks);
        else
            k_end = kt + k_inc(ks) + 1;
        end

        for k = k_start:k_end
            % angle perturbation
    %         if k_start==1&&k==1;bus_v(2,1)=abs(bus_v(2,1))*exp(1i*(angle(bus_v(2,1))+2));;end
            % step 3a: network solution
            % mach_ref(k) = mac_ang(syn_ref,k);
            mach_ref(k) = 0;
            pmech(:,k+1) = pmech(:,k);
            tmig(:,k+1) = tmig(:,k);

            if n_conv~=0;cur_ord(:,k+1) = cur_ord(:,k);end

            flag = 1;
            timestep = int2str(k);
            % network-machine interface
            mac_ind(0,k,bus_sim,flag);
            mac_igen(0,k,bus_sim,flag);
            mac_sub(0,k,bus_sim,flag);
            mac_tra(0,k,bus_sim,flag);
            mac_em(0,k,bus_sim,flag);
            mdc_sig(t(k),k);
            dc_cont(0,k,10*(k-1)+1,bus_sim,flag);
            % Calculate current injections and bus voltages and angles
            if k >= sum(k_inc(1:3))+1
                % fault cleared
                line_sim = line_pf2;
                bus_sim = bus_pf2;
                bus_int = bus_intpf2;
                Y1 = Y_gpf2;
                Y2 = Y_gncpf2;
                Y3 = Y_ncgpf2;
                Y4 = Y_ncpf2;
                Vr1 = V_rgpf2; 
                Vr2 = V_rncpf2;
                bo = bopf2;
                h_sol = i_simu(k,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);
            elseif k >=sum(k_inc(1:2))+1
                % near bus cleared
                line_sim = line_pf1;
                bus_sim = bus_pf1;
                bus_int = bus_intpf1;
                Y1 = Y_gpf1;
                Y2 = Y_gncpf1;
                Y3 = Y_ncgpf1;
                Y4 = Y_ncpf1;
                Vr1 = V_rgpf1; 
                Vr2 = V_rncpf1;
                bo = bopf1;
                h_sol = i_simu(k,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);   
            elseif k>=k_inc(1)+1
                % fault applied
                line_sim = line_f;
                bus_sim = bus_f;
                bus_int = bus_intf;
                Y1 = Y_gf;
                Y2 = Y_gncf;
                Y3 = Y_ncgf;
                Y4 = Y_ncf;
                Vr1 = V_rgf; 
                Vr2 = V_rncf;
                bo = bof;
                h_sol = i_simu(k,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);     
            elseif k<k_inc(1)+1
                % pre fault
                line_sim = line;
                bus_sim = bus;
                bus_int = bus_intprf;
                Y1 = Y_gprf;
                Y2 = Y_gncprf;
                Y3 = Y_ncgprf;
                Y4 = Y_ncprf;
                Vr1 = V_rgprf; 
                Vr2 = V_rncprf;
                bo = boprf;
                h_sol = i_simu(k,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);  
            end
          % HVDC
          
          dc_cont(0,k,10*(k-1)+1,bus_sim,flag);
          % network interface for control models
          dpwf(0,k,bus_sim,flag);
          mexc_sig(t(k),k);
          smpexc(0,k,bus_sim,flag);
          smppi(0,k,bus_sim,flag);
          exc_st3(0,k,bus_sim,flag);
          exc_dc12(0,k,bus_sim,flag);
          mtg_sig(t(k),k);
          tg(0,k,bus_sim,flag);
          tg_hydro(0,k,bus_sim,flag);
          if n_dcud~=0
             % set the new line currents
             for jj=1:n_dcud
                l_num = svc_dc{jj,3};svc_num = svc_dc{jj,2};
                from_bus = bus_int(line_sim(l_num,1)); to_bus=bus_int(line_sim(l_num,2));
                svc_bn = bus_int(svc_con(svc_num,2));
                V1 = bus_v(from_bus,k);
                V2 = bus_v(to_bus,k);
                R = line_sim(l_num,3);X=line_sim(l_num,4);B=line_sim(l_num,5);tap = line_sim(l_num,6);phi = line_sim(l_num,7);
                [l_if,l_it] = line_cur(V1,V2,R,X,B,tap,phi);
                if svc_bn == from_bus; d_sig(jj,k)=abs(l_if);elseif svc_bn==to_bus;d_sig(jj,k)=abs(l_it); end
             end
          end
          if n_tcscud~=0
             % set the new bus voltages
             for jj=1:n_tcscud
                b_num = tcsc_dc{jj,3};tcsc_num = tcsc_dc{jj,2};
                td_sig(jj,k)=abs(bus_v(bus_int(b_num),k));
             end
          end

%           i_plot=k-plot_now;
%           if i_plot == 10
%              plot_now = k;
%              v_p(1:k)=abs(bus_v(bus_idx(1),1:k));
%              % plot the voltage of the faulted bus
%              plot(t(1:k),v_p(1:k),'r')
%              title(['Voltage Magnitude at ' num2str(bus(bus_idx(1),1)) ' ' dfile]);
%              xlabel('time (s)');
%              drawnow
%           end
          % step 3b: compute dynamics and integrate
          flag = 2;
          sys_freq(k) = 1.0;
          mpm_sig(t(k),k);
          mac_ind(0,k,bus_sim,flag);
          mac_igen(0,k,bus_sim,flag);
          mac_sub(0,k,bus_sim,flag); 
          mac_tra(0,k,bus_sim,flag);
          mac_em(0,k,bus_sim,flag);
          dpwf(0,k,bus_sim,flag);
          pss(0,k,bus_sim,flag);
          mexc_sig(t(k),k);
          smpexc(0,k,bus_sim,flag);
          smppi(0,k,bus_sim,flag);
          exc_st3(0,k,bus_sim,flag);
          exc_dc12(0,k,bus_sim,flag);
          mtg_sig(t(k),k);
          tg(0,k,bus_sim,flag);
          tg_hydro(0,k,bus_sim,flag);

          % integrate dc at ten times rate
          mdc_sig(t(k),k);
          if n_conv~=0
             hdc_sol = h_sol/10;
             for kk = 1:10
                kdc=10*(k-1)+kk;
                [xdcr_dc(:,kdc:kdc+1),dxdcr_dc(:,kdc:kdc+1),xdci_dc(:,kdc:kdc+1),dxdci_dc(:,kdc:kdc+1)] = ...
                   dc_sim(k,kk,dcr_dc,dci_dc,xdcr_dc(:,kdc),xdci_dc(:,kdc),bus_sim,hdc_sol); 
             end
          else
             dc_cont(0,k,k,bus_sim,2);
             dc_line(0,k,k,bus_sim,2);
          end

          j = k+1;
    %       theta(1,1)=theta(1,1)+pi;
          % following statements are predictor steps
          mac_ang(:,j) = mac_ang(:,k) + h_sol*dmac_ang(:,k); 
          mac_spd(:,j) = mac_spd(:,k) + h_sol*dmac_spd(:,k);
          edprime(:,j) = edprime(:,k) + h_sol*dedprime(:,k);
          eqprime(:,j) = eqprime(:,k) + h_sol*deqprime(:,k);
          psikd(:,j) = psikd(:,k) + h_sol*dpsikd(:,k);
          psikq(:,j) = psikq(:,k) + h_sol*dpsikq(:,k);
          Efd(:,j) = Efd(:,k) + h_sol*dEfd(:,k);
          V_R(:,j) = V_R(:,k) + h_sol*dV_R(:,k);
          V_As(:,j) = V_As(:,k) + h_sol*dV_As(:,k);
          R_f(:,j) = R_f(:,k) + h_sol*dR_f(:,k);
          V_TR(:,j) = V_TR(:,k) + h_sol*dV_TR(:,k);
          sdpw1(:,j) = sdpw1(:,k) + h_sol*dsdpw1(:,k);
          sdpw2(:,j) = sdpw2(:,k) + h_sol*dsdpw2(:,k);
          sdpw3(:,j) = sdpw3(:,k) + h_sol*dsdpw3(:,k);
          sdpw4(:,j) = sdpw4(:,k) + h_sol*dsdpw4(:,k);
          sdpw5(:,j) = sdpw5(:,k) + h_sol*dsdpw5(:,k);
          sdpw6(:,j) = sdpw6(:,k) + h_sol*dsdpw6(:,k);
          pss1(:,j) = pss1(:,k) + h_sol*dpss1(:,k);
          pss2(:,j) = pss2(:,k) + h_sol*dpss2(:,k);
          pss3(:,j) = pss3(:,k) + h_sol*dpss3(:,k);
          tg1(:,j) = tg1(:,k) + h_sol*dtg1(:,k);
          tg2(:,j) = tg2(:,k) + h_sol*dtg2(:,k);
          tg3(:,j) = tg3(:,k) + h_sol*dtg3(:,k);
          tg4(:,j) = tg4(:,k) + h_sol*dtg4(:,k);
          tg5(:,j) = tg5(:,k) + h_sol*dtg5(:,k);
          vdp(:,j) = vdp(:,k) + h_sol*dvdp(:,k);
          vqp(:,j) = vqp(:,k) + h_sol*dvqp(:,k);
          slip(:,j) = slip(:,k) + h_sol*dslip(:,k);
          vdpig(:,j) = vdpig(:,k) + h_sol*dvdpig(:,k);
          vqpig(:,j) = vqpig(:,k) + h_sol*dvqpig(:,k);
          slig(:,j) = slig(:,k) + h_sol*dslig(:,k);
          B_cv(:,j) = B_cv(:,k) + h_sol*dB_cv(:,k);
          B_con(:,j) = B_con(:,k) + h_sol*dB_con(:,k);
          xsvc_dc(:,j) = xsvc_dc(:,k) + h_sol* dxsvc_dc(:,k);
          B_tcsc(:,j) = B_tcsc(:,k) + h_sol*dB_tcsc(:,k);
          xtcsc_dc(:,j) = xtcsc_dc(:,k) + h_sol* dxtcsc_dc(:,k);
          lmod_st(:,j) = lmod_st(:,k) + h_sol*dlmod_st(:,k);
          rlmod_st(:,j) = rlmod_st(:,k)+h_sol*drlmod_st(:,k);
          flag = 1;
          % mach_ref(j) = mac_ang(syn_ref,j);
          mach_ref(j) = 0;
          % perform network interface calculations again with predicted states
          mpm_sig(t(j),j);
          mac_ind(0,j,bus_sim,flag);
          mac_igen(0,j,bus_sim,flag);
          mac_sub(0,j,bus_sim,flag);
          mac_tra(0,j,bus_sim,flag);
          mac_em(0,j,bus_sim,flag);
          % assume Vdc remains unchanged for first pass through dc controls interface
          mdc_sig(t(j),j);
          dc_cont(0,j,10*(j-1)+1,bus_sim,flag);

          % Calculate current injections and bus voltages and angles
          if j >= sum(k_inc(1:3))+1
             % fault cleared
             bus_sim = bus_pf2;
             bus_int = bus_intpf2;
             Y1 = Y_gpf2;
             Y2 = Y_gncpf2;
             Y3 = Y_ncgpf2;
             Y4 = Y_ncpf2;
             Vr1 = V_rgpf2; 
             Vr2 = V_rncpf2;
             bo = bopf2;
             h_sol = i_simu(j,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);     
          elseif j >=sum(k_inc(1:2))+1
             % near bus cleared
             bus_sim = bus_pf1;
             bus_int = bus_intpf1;
             Y1 = Y_gpf1;
             Y2 = Y_gncpf1;
             Y3 = Y_ncgpf1;
             Y4 = Y_ncpf1;
             Vr1 = V_rgpf1; 
             Vr2 = V_rncpf1;
             bo = bopf1;
             h_sol = i_simu(j,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);   
          elseif j>=k_inc(1)+1
             % fault applied
             bus_sim = bus_f;
             bus_int = bus_intf;
             Y1 = Y_gf;
             Y2 = Y_gncf;
             Y3 = Y_ncgf;
             Y4 = Y_ncf;
             Vr1 = V_rgf; 
             Vr2 = V_rncf;
             bo = bof;
             h_sol = i_simu(j,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);     
          elseif k<k_inc(1)+1
             % pre fault
             bus_sim = bus;
             bus_int = bus_intprf;
             Y1 = Y_gprf;
             Y2 = Y_gncprf;
             Y3 = Y_ncgprf;
             Y4 = Y_ncprf;
             Vr1 = V_rgprf; 
             Vr2 = V_rncprf;
             bo = boprf;
             h_sol = i_simu(j,ks,k_inc,h,bus_sim,Y1,Y2,Y3,Y4,Vr1,Vr2,bo);     
          end
          vex(:,j)=vex(:,k);
          cur_ord(:,j) = cur_ord(:,k);
          % calculate the new value of bus angles rectifier user defined control
          if ndcr_ud~=0
             tot_states=0;
             for jj = 1:ndcr_ud
                b_num1 = dcr_dc{jj,3};b_num2 = dcr_dc{jj,4};conv_num = dcr_dc{jj,2};
                angdcr(jj,j)=theta(bus_int(b_num1),j)-theta(bus_int(b_num2),j);
                dcrd_sig(jj,j)=angdcr(jj,j);
                st_state = tot_states+1; dcr_states = dcr_dc{jj,7}; tot_states = tot_states+dcr_states; 
                ydcrmx=dcr_dc{jj,5};ydcrmn = dcr_dc{jj,6};
                dcr_dsig(jj,j) = ...
                   dcr_sud(jj,j,flag,dcr_dc{jj,1},dcrd_sig(jj,j),ydcrmx,ydcrmn,xdcr_dc(st_state:tot_states,10*(j-1)+1));
             end
          end
          if ndci_ud~=0
             % calculate the new value of bus angles inverter user defined control
             for jj = 1:ndci_ud
                tot_states=0;
                b_num1 = dci_dc{jj,3};b_num2 = dci_dc{jj,4};conv_num = dci_dc{jj,2};
                angdci(jj,j)=theta(bus_int(b_num1),j)-theta(bus_int(b_num2),j);
                dcid_sig(jj,j)=(angdci(jj,j)-angdci(jj,k))/(t(j)-t(k));
                st_state = tot_states+1; dci_states = dci_dc{jj,7}; tot_states = tot_states+dci_states; 
                ydcimx=dci_dc{jj,5};ydcimn = dci_dc{jj,6};
                dci_dsig(jj,j) = ...
                   dci_sud(jj,j,flag,dci_dc{jj,1},dcid_sig(jj,j),ydcimx,ydcimn,xdci_dc(st_state:tot_states,10*(j-1)+1));
             end
          end 
          dc_cont(0,j,10*(j-1)+1,bus_sim,flag);
          dpwf(0,j,bus_sim,flag);
          pss(0,j,bus_sim,flag);
          mexc_sig(t(j),j);
          smpexc(0,j,bus_sim,flag);
          smppi(0,j,bus_sim,flag);
          exc_st3(0,j,bus_sim,flag);
          exc_dc12(0,j,bus_sim,flag);
          tg(0,j,bus_sim,flag); 
          tg_hydro(0,j,bus_sim,flag);
          if n_dcud~=0
             % set the new line currents
             for jj=1:n_dcud
                l_num = svc_dc{jj,3};svc_num = svc_dc{jj,2};
                from_bus = bus_int(line_sim(l_num,1)); to_bus=bus_int(line_sim(l_num,2));
                svc_bn = bus_int(svc_con(svc_num,2));
                V1 = bus_v(from_bus,j);
                V2 = bus_v(to_bus,j);
                R = line_sim(l_num,3);X=line_sim(l_num,4);B=line_sim(l_num,5);tap = line_sim(l_num,6);phi = line_sim(l_num,7);
                [l_if,l_it] = line_cur(V1,V2,R,X,B,tap,phi);
                if svc_bn == from_bus; 
                   d_sig(jj,j)=abs(l_if);            
                elseif svc_bn==to_bus;
                   d_sig(jj,j)=abs(l_it);
                end
             end
          end
          if n_tcscud~=0
             % set the new line currents
             for jj=1:n_tcscud
                b_num = tcsc_dc{jj,3};tcsc_num = tcsc_dc{jj,2};
                td_sig(jj,j)=abs(bus_v(bus_int(b_num),j));
             end
          end

          flag = 2;
          mac_ind(0,j,bus_sim,flag);
          mac_igen(0,j,bus_sim,flag);
          mac_sub(0,j,bus_sim,flag);
          mac_tra(0,j,bus_sim,flag);
          mac_em(0,j,bus_sim,flag);
          dpwf(0,j,bus_sim,flag);
          pss(0,j,bus_sim,flag);
          mexc_sig(t(j),j);
          smpexc(0,j,bus_sim,flag);
          smppi(0,j,bus_sim,flag);
          exc_st3(0,j,bus_sim,flag);
          exc_dc12(0,j,bus_sim,flag);
          mtg_sig(t(j),j);
          tg(0,j,bus_sim,flag);
          tg_hydro(0,j,bus_sim,flag);
          
          if n_conv~=0
             hdc_sol = h_sol/10;
             for kk = 1:10
                jdc=10*(j-1)+kk;
                [xdcr_dc(:,jdc:jdc+1),dxdcr_dc(:,jdc:jdc+1),xdci_dc(:,jdc:jdc+1),dxdci_dc(:,jdc:jdc+1)] = ...
                   dc_sim(j,kk,dcr_dc,dci_dc,xdcr_dc(:,jdc),xdci_dc(:,jdc),bus_sim,hdc_sol); 
             end
          else
             dc_cont(0,j,j,bus_sim,2);
             dc_line(0,j,j,bus_sim,2);
          end
          % following statements are corrector steps
          mac_ang(:,j) = mac_ang(:,k) +...
             h_sol*(dmac_ang(:,k)+dmac_ang(:,j))/2.;
          mac_spd(:,j) = mac_spd(:,k) +...
             h_sol*(dmac_spd(:,k)+dmac_spd(:,j))/2.;
          edprime(:,j) = edprime(:,k) +...
             h_sol*(dedprime(:,k)+dedprime(:,j))/2.;
          eqprime(:,j) = eqprime(:,k) +...
             h_sol*(deqprime(:,k)+deqprime(:,j))/2.;
          psikd(:,j) = psikd(:,k) +...
             h_sol*(dpsikd(:,k)+dpsikd(:,j))/2.;
          psikq(:,j) = psikq(:,k) +...
             h_sol*(dpsikq(:,k)+dpsikq(:,j))/2.;
          Efd(:,j) = Efd(:,k) +...
             h_sol*(dEfd(:,k)+dEfd(:,j))/2.;
          V_R(:,j) = V_R(:,k) +...
             h_sol*(dV_R(:,k)+dV_R(:,j))/2.;
          V_As(:,j) = V_As(:,k) +...
             h_sol*(dV_As(:,k)+dV_As(:,j))/2.;
          R_f(:,j) = R_f(:,k) +...
             h_sol*(dR_f(:,k)+dR_f(:,j))/2.;
          V_TR(:,j) = V_TR(:,k) +...
             h_sol*(dV_TR(:,k)+dV_TR(:,j))/2.;
          sdpw11(:,j) = sdpw1(:,k) +h_sol*(dsdpw1(:,k)+dsdpw1(:,j))/2.;
          sdpw12(:,j) = sdpw2(:,k) +h_sol*(dsdpw2(:,k)+dsdpw2(:,j))/2.;
          sdpw13(:,j) = sdpw3(:,k) +h_sol*(dsdpw3(:,k)+dsdpw3(:,j))/2.;
          sdpw14(:,j) = sdpw4(:,k) +h_sol*(dsdpw4(:,k)+dsdpw4(:,j))/2.;
          sdpw15(:,j) = sdpw5(:,k) +h_sol*(dsdpw5(:,k)+dsdpw5(:,j))/2.;
          sdpw16(:,j) = sdpw6(:,k) +h_sol*(dsdpw6(:,k)+dsdpw6(:,j))/2.;
          pss1(:,j) = pss1(:,k) +h_sol*(dpss1(:,k)+dpss1(:,j))/2.;
          pss2(:,j) = pss2(:,k) +h_sol*(dpss2(:,k)+dpss2(:,j))/2.;
          pss3(:,j) = pss3(:,k) +h_sol*(dpss3(:,k)+dpss3(:,j))/2.;
          tg1(:,j) = tg1(:,k) + h_sol*(dtg1(:,k) + dtg1(:,j))/2.;
          tg2(:,j) = tg2(:,k) + h_sol*(dtg2(:,k) + dtg2(:,j))/2.;
          tg3(:,j) = tg3(:,k) + h_sol*(dtg3(:,k) + dtg3(:,j))/2.;
          tg4(:,j) = tg4(:,k) + h_sol*(dtg4(:,k) + dtg4(:,j))/2.;
          tg5(:,j) = tg5(:,k) + h_sol*(dtg5(:,k) + dtg5(:,j))/2.;
          vdp(:,j) = vdp(:,k) + h_sol*(dvdp(:,j) + dvdp(:,k))/2.;
          vqp(:,j) = vqp(:,k) + h_sol*(dvqp(:,j) + dvqp(:,k))/2.;
          slip(:,j) = slip(:,k) + h_sol*(dslip(:,j) + dslip(:,k))/2.;
          vdpig(:,j) = vdpig(:,k) + h_sol*(dvdpig(:,j) + dvdpig(:,k))/2.;
          vqpig(:,j) = vqpig(:,k) + h_sol*(dvqpig(:,j) + dvqpig(:,k))/2.;
          slig(:,j) = slig(:,k) + h_sol*(dslig(:,j) + dslig(:,k))/2.;
          B_cv(:,j) = B_cv(:,k) + h_sol*(dB_cv(:,j) + dB_cv(:,k))/2.;
          B_con(:,j) = B_con(:,k) + h_sol*(dB_con(:,j) + dB_con(:,k))/2.;
          xsvc_dc(:,j) = xsvc_dc(:,k) + h_sol*(dxsvc_dc(:,j) + dxsvc_dc(:,k))/2.;
          B_tcsc(:,j) = B_tcsc(:,k) + h_sol*(dB_tcsc(:,j) + dB_tcsc(:,k))/2.;
          xtcsc_dc(:,j) = xtcsc_dc(:,k) + h_sol*(dxtcsc_dc(:,j) + dxtcsc_dc(:,k))/2.;
          lmod_st(:,j) = lmod_st(:,k) + h_sol*(dlmod_st(:,j) + dlmod_st(:,k))/2.;
          rlmod_st(:,j) = rlmod_st(:,k) + h_sol*(drlmod_st(:,j) + drlmod_st(:,k))/2.;
       end
       kt = kt + k_inc(ks);
       ks = ks+1;
    end

    % process phase angle
    theta_1=theta(1:39,:);
    for ii=1:39
        for jj=1:length(theta_1(1,:))-1
            if theta(ii,jj) > 0 && theta(ii,jj+1) < 0
                theta_1(ii,jj+1:end)=theta_1(ii,jj+1:end)+2*pi;
            end
        end
    end
    bus_theta(i,:,:)=theta_1;
    bus_voltage(i,:,:)=abs(bus_v(1:39,:));
    gen_theta(i,:,:)=mac_ang;
    gen_speed(i,:,:)=mac_spd;
    ind_slip(i,:,:)=slip;
    et = toc;
    ets = num2str(et);
    disp(['(',num2str(N),',',num2str(i),',',num2str(nn),'/',num2str(m),') elapsed time = ',ets,'s'])
end
clear
clc