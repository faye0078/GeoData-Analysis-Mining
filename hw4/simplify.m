file = dir(fullfile('C:\Users\Faye\code\mfile\date\data_recons','*.txt'));
[length,width]=size(file);
flag=1;
flag2=1;
sum=0;
sumzong=0;
 
for numberfile=1:length
    shipdata=newReadFile(['C:\Users\Faye\code\mfile\date\data_recons\',file(numberfile).name],7);
    [m,n]=size(shipdata);
    sumzong=sumzong+m;
    xpoints=zeros(1,m);
    for i=1:m
        ypoint=str2num(shipdata{i,5});
        xpoint=str2num(shipdata{i,4});
        xpoints(1,i)=xpoint;
        ypoints(1,i)=ypoint;
    end
    num = size(xpoints,2);
    head = [xpoints(1), ypoints(1)] ;
    tail = [xpoints(end), ypoints(end)] ;
    mindist = 0.001;
    max_dist = 0;
    max_dist_i = 0;
    curve = [head; tail];
    pnt_index = [1, num];
    while(1)
        curve_num = size(curve,1);
        add_new_pnt = 0;
        for nx = 1:curve_num-1
            cur_pnt = curve(nx,:);
            next_pnt = curve(nx+1,:);
            pnt_d = next_pnt - cur_pnt ;
            if ~isreal(pnt_d(1))||~isreal(pnt_d(2))
                flag=0;
                number=size(pnt_index,2);
                nshipdata=cell(number,7);
                for i=1:number
                    nshipdata(i,:)=shipdata(pnt_index(i),:);
                end
                writecell(nshipdata,['C:\Users\Faye\code\mfile\date\data_smp1m\',file(numberfile).name]);
                break;
            end
            
            th = atan2(pnt_d(2), pnt_d(1));
            angle = th*180/pi;
            k = tan(th);
            b = cur_pnt(2) - k*cur_pnt(1);
            k2 = k*k;
            deno = sqrt(1 + k *k) ;
            max_dist = 0;
            pnt_index(nx);
            pnt_index(nx+1);
            for i= pnt_index(nx) : pnt_index(nx+1)
                dist = abs(ypoints(i) - k*xpoints(i) - b)/deno ;
                
                if(dist> max_dist)
                    max_dist = dist;
                    max_dist_i = i;
                end
            end
            if max_dist_i==0
                number=size(pnt_index,2);
                nshipdata=cell(number,7);
                for i=1:number
                    nshipdata(i,:)=shipdata(pnt_index(i),:);
                end
                writecell(nshipdata,['C:\Users\Faye\code\mfile\date\data_smp1m\',file(numberfile).name]);
                break;
            end
            far_pnt = [xpoints(max_dist_i), ypoints(max_dist_i)];
 
            if(max_dist > mindist)
                curve = [curve(1:nx,:); far_pnt; curve(nx+1:end,:)];
                pnt_index = [pnt_index(1:nx), max_dist_i, pnt_index(nx+1:end)];
                
                add_new_pnt = 1;
            end
 
            if(0 == add_new_pnt)
                flag2=0;
                break;    
            end
        end
        if flag2==0
            break;
        end
        if flag==0
            break;
        end
    end
    number=size(pnt_index,2);
    sum=number+sum;
    nshipdata=cell(number,7);
    for i=1:number
        nshipdata(i,:)=shipdata(pnt_index(i),:);
    end
    writecell(nshipdata,['C:\Users\Faye\code\mfile\date\data_smp1m\',file(numberfile).name]);
end
result=sum/sumzong;