N_block=100;
for c_block=1:N_block
     GeneTraData;
     save(['.\Input_' num2str(c_block) '.mat'],'Input');
     save(['.\Output_' num2str(c_block) '.mat'],'Output');
     clear;
end