clc;
clear all;
 
file = dir(fullfile('C:\Users\Faye\code\mfile\date\data_unoise','*.txt'));
[length,width]=size(file);
for numberfile=1:length
    
    shipdata=newReadFile(['C:\Users\Faye\code\mfile\date\data_unoise\',file(numberfile).name],7);
    
    [m,n]=size(shipdata);
    number=0;
    xpoints=zeros(1,7);
    ypoints=zeros(1,7);
    
    
    for i=2:m
        
        pointy=[str2num(shipdata{i,4}),str2num(shipdata{i,5})];
        pointx=[str2num(shipdata{i-1,4}),str2num(shipdata{i-1,5})];
        
        if(distance(pointx,pointy)>1)
            s=(str2num(shipdata{i,2})-str2num(shipdata{i-1,2}))/(3/60);
            b=int8(s);
            n=double(b);
            xx=str2num(shipdata{i,4})-str2num(shipdata{i-1,4});
            yy=str2num(shipdata{i,5})-str2num(shipdata{i-1,5});
            k=i-1;
            for j=1:n
                
                x0=str2num(shipdata{i-1,4})+(j*xx)/s;
                y0=str2num(shipdata{i-1,5})+(j*yy)/s;
                time=num2str(str2num(shipdata{i-1,2})+(j*(str2num(shipdata{i,2})-str2num(shipdata{i-1,2})))/n);
                addata={shipdata{1,1},time,shipdata{1,3},num2str(x0),num2str(y0),shipdata{i,6},shipdata{i,7}};
                
                shipdata=[shipdata(1:k,:);addata;shipdata(k+1:end,:)];
                k=k+1;
            end
        end
    end
    nm=size(shipdata,1);
    for i=m+1:nm
        pointy=[str2num(shipdata{i,4}),str2num(shipdata{i,5})];
        pointx=[str2num(shipdata{i-1,4}),str2num(shipdata{i-1,5})];
        d=distance(pointx,pointy);
        if(d>1)
            s=(str2num(shipdata{i,2})-str2num(shipdata{i-1,2}))/(3/60);
            b=int8(s);
            n=double(b);
            xx=str2num(shipdata{i,4})-str2num(shipdata{i-1,4});
            yy=str2num(shipdata{i,5})-str2num(shipdata{i-1,5});
            k=i-1;
            for j=1:n
                
                x0=str2num(shipdata{i-1,4})+(j*xx)/s;
                y0=str2num(shipdata{i-1,5})+(j*yy)/s;
                time=num2str(str2num(shipdata{i-1,2})+(j*(str2num(shipdata{i,2})-str2num(shipdata{i-1,2})))/n);
                addata={shipdata{1,1},time,shipdata{1,3},num2str(x0),num2str(y0),shipdata{i,6},shipdata{i,7}};
                shipdata=[shipdata(1:k,:);addata;shipdata(k+1:end,:)];
                k=k+1;
            end
        end
    end
    
    writecell(shipdata,['C:\Users\Faye\code\mfile\date\data_recons\',file(numberfile).name]);
    
end
function d=distance(x,y)
d=sqrt((x(1)-y(1))^2+(x(2)-y(2))^2);
end
function f0 = Lagrange(x,y,x0)
 
%???????Lagrange?????f?????????f????x0????f0
syms t;
n = length(x);
f = 0.0;
for i = 1:n
    l = y(i);
    for j = 1:i-1
        l = l*(t-x(j))/(x(i)-x(j));
    end
    for j = i+1:n
        l = l*(t-x(j))/(x(i)-x(j));
    end
    f = f + l;
    simplify(f);
    if(i==n)
        f0 = subs(f,'t',x0);
        f = collect(f);
        f = vpa(f,6);
    end
end
end
