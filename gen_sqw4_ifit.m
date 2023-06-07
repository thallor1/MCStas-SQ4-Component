sq = sw_model('squareAF',2,0);                  % create the SW object (using SpinW)
s=sqw_spinw(sq);                                        % create the Model
qh=linspace(-1.5,1.5,100);qk=qh; ql=qh'; w=linspace(0.01,10,25);
f=iData(s,s.p,qh,qk,ql,w); plot3(log(f(:,:,1,:)));         % evaluate and plot