clc;
clear all;
 
file = dir(fullfile('C:\Users\Faye\code\mfile\date\newdata_xy','*.txt'));
 
[length,width]=size(file);
for numberfile=1:length
    
    shipdata=newReadFile(['C:\Users\Faye\code\mfile\date\newdata_xy\',file(numberfile).name],7);
    [m,n]=size(shipdata);
    number=0;
    
    for i=2:m-1
        pointy=[str2num(shipdata{i,4}),str2num(shipdata{i,5})];
        pointx=[str2num(shipdata{i-1,4}),str2num(shipdata{i-1,5})];
        pointz=[str2num(shipdata{i+1,4}),str2num(shipdata{i+1,5})];
        
        v1=1.852*(str2num(shipdata{i,6})+str2num(shipdata{i-1,6}))/2;
        v2=1.852*(str2num(shipdata{i,6})+str2num(shipdata{i+1,6}))/2;
        averspd1=distance(pointx,pointy)/(str2num(shipdata{i,2})-str2num(shipdata{i-1,2}));
        averspd2=distance(pointy,pointz)/(str2num(shipdata{i+1,2})-str2num(shipdata{i,2}));
        if (~strcmp(num2str(v1),'0'))&&(~strcmp(num2str(v2),'0'))&&(~strcmp(num2str(v1),'N/A'))&&(~strcmp(num2str(v2),'N/A'))
            if (averspd1>1.5*v1)&&(averspd2>1.5*v2)
                number=number+1;
                n=i;
                shipdata{i,4}=string((str2num(shipdata{i-1,4})+str2num(shipdata{i+1,4}))/2);
                shipdata{i,5}=string((str2num(shipdata{i-1,5})+str2num(shipdata{i+1,5}))/2);
            end
        end
    end
    
    writecell(shipdata,['C:\Users\Faye\code\mfile\date\data_unoise\',file(numberfile).name]);
    
end
function d=distance(x,y)
d=sqrt((x(1)-y(1))^2+(x(2)-y(2))^2);
end
