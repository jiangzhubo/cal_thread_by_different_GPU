import sys
import os, subprocess
import atexit
import time
import psutil
import numpy as np
def get_gpu_info():
   nvidia_memory ={}
   intel_amd_gpu_memory = {}
   gpu_count = 0 
   nvidiaMemory = 0 
   intelAmdGpuMemory = 0 
   res = subprocess.Popen('grep -i memory /var/log/Xorg.1.log',shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True)
   result = res.stdout.readlines()
   if len(result) ==0:
        res = subprocess.Popen('grep -i memory /var/log/Xorg.0.log',shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True)
        result = res.stdout.readlines()
   for index,gpu_i in enumerate(result) :  
     temp = str(gpu_i, encoding = "utf-8").split(' ')
     temp = [tem.split('(')[0] for tem in temp ]
     for i in  temp:
         has_nvidia = 0
         if i in ['nvidia','Nvidia','NVIDIA']:
             has_nvidia = 1
             break
     if has_nvidia == 1: 
         for indx,x in  enumerate(temp):
            if x in ['memory:','Memory:'] and float(temp[indx+1]) > 0 :
               nvidia_memory[index] = float(temp[indx+1])/1024
             
     elif has_nvidia == 0:
          for indx,x in  enumerate(temp):
            if x in ['memory:','Memory:'] and float(temp[indx+1]) > 0:
               intel_amd_gpu_memory[index] = float(temp[indx+1])/1024
   nvidiaMemory=sorted(nvidia_memory.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
   intelAmdGpuMemory=sorted(intel_amd_gpu_memory.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
   print(nvidiaMemory,intelAmdGpuMemory)
   gpu_count = len(nvidia_memory)+ len(intel_amd_gpu_memory)
   return  gpu_count,nvidiaMemory,intelAmdGpuMemory

#function of Get CPU State
def getCPUstate():
    cpu_core  = psutil.cpu_count()
    phymem = psutil.virtual_memory()
    buffers = getattr(psutil, 'phymem_buffers', lambda: 0)()
    cached = getattr(psutil, 'cached_phymem', lambda: 0)()
    used = phymem.total - (phymem.free + buffers + cached)
    total_cpu_memory = int(phymem.total / 1024 / 1024)
    single_core_memory = total_cpu_memory / cpu_core
    all_cores_memory_left = [single_core_memory - single_core_usage * single_core_memory/100.0 for single_core_usage in psutil.cpu_percent(1,percpu=True)]
    line = "total CPU Memory: %5s%% %6s/%s" % (
        phymem.percent,
        str(int(used / 1024 / 1024)) + "M",
        str(int(phymem.total / 1024 / 1024)) + "M"  )
    print(line)
    print('every core current memory left:',all_cores_memory_left)
    free_cpu_memory = int(phymem.total / 1024 / 1024) -  int(used / 1024 / 1024)
    return free_cpu_memory,all_cores_memory_left,cpu_core

def get_valid_core(all_cpu_cores_memory_left,size_for_one_thread):
    valid_core = 0 
    for single_core in all_cpu_cores_memory_left:
             if single_core > size_for_one_thread:
                 valid_core +=1
    
    return valid_core
def decide_thread_number():
    '''
       output: how much threads for specific gpu, if no gpu, threads[0] means will be results.
    '''
    default_using_gpu = 800 # when load models, this is basic size of how much gpu memory that models use
    default_using_cpu = 1200 # when load models, this is basic size of how much gpu memory that models use
    max_using_cpu_for_one_thread = 1200 # when process dicom without GPU, how much cpu that models use when process one thread  
    global  size_for_one_thread 
    size_for_one_thread  = 1000   # when process dicom with GPU, how much cpu and GPU that models use when process one thread
    number_gpu = 0 
    free_cpu_memory,all_cpu_cores_memory_left,cpu_core =  getCPUstate()
    threads = {}
    valid_core = get_valid_core(all_cpu_cores_memory_left,max_using_cpu_for_one_thread) /2
    number_gpu,nvidia_gpus,other_gpus = get_gpu_info()
    idx,ind = 0, 0 
    if number_gpu ==0: # No gpu in machine
       print('we can not find any gpu')
       valid_thread_for_whole_cpu = int((free_cpu_memory - default_using_cpu) / max_using_cpu_for_one_thread)
       print(valid_core,valid_thread_for_whole_cpu)
       threads[0] =['UNKNOWN', min(valid_thread_for_whole_cpu,valid_core)]
    elif number_gpu > 0:
     if len(nvidia_gpus) > 0:
        for idx, [gpu_index,gpu_memory] in enumerate(nvidia_gpus):
           threads_for_gpu = max(int((gpu_memory- default_using_gpu) /size_for_one_thread ),0) # if start another GPU, machine has to offer space for loading moders
           free_cpu_memory -= default_using_cpu # delete the size of models
           threads_for_cpu = max(int((free_cpu_memory)/size_for_one_thread),0)
           thread_for_final =  min(threads_for_cpu,threads_for_gpu)
           threads[idx] =[gpu_index, min(thread_for_final,max(valid_core,0))]
           valid_core -= thread_for_final
           free_cpu_memory -= thread_for_final * size_for_one_thread
       
     if len(other_gpus)  > 0:
        for ind, [gpu_index,gpu_memory] in enumerate(other_gpus):
           threads_for_gpu = max(int((gpu_memory- default_using_gpu) /size_for_one_thread ),0) # if start another GPU, machine has to offer space for loading moders
           free_cpu_memory -= default_using_cpu # delete the size of models
           threads_for_cpu = max(int((free_cpu_memory)/size_for_one_thread),0)
           thread_for_final =  min(threads_for_cpu,threads_for_gpu)
           threads[ind+idx+1] =[gpu_index, min(thread_for_final,max(valid_core,0))]
           valid_core -= thread_for_final
           free_cpu_memory -= thread_for_final * size_for_one_thread


    
    return threads
 
              
#example              
print(decide_thread_number())  
