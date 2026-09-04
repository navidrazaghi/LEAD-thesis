#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef struct { uint32_t sType; const void* pNext; const char* pAppName;
  uint32_t appVer; const char* pEngName; uint32_t engVer; uint32_t apiVer; } AppInfo;
typedef struct { uint32_t sType; const void* pNext; uint32_t flags;
  const AppInfo* pApp; uint32_t layerCount; const char* const* layers;
  uint32_t extCount; const char* const* exts; } InstInfo;
extern int vkCreateInstance(const InstInfo*, const void*, void**);
extern int vkEnumeratePhysicalDevices(void*, uint32_t*, void**);
extern void vkGetPhysicalDeviceProperties(void*, void*);
int main(void){
  AppInfo app; memset(&app,0,sizeof app); app.sType=0; app.apiVer=(1<<22);
  InstInfo ci; memset(&ci,0,sizeof ci); ci.sType=1; ci.pApp=&app;
  void* inst=0; int r=vkCreateInstance(&ci,0,&inst);
  if(r!=0){ printf("vkCreateInstance failed: %d\n", r); return 1; }
  uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,0);
  printf("physical devices: %u\n", n);
  if(n>8) n=8;
  void* devs[8]; vkEnumeratePhysicalDevices(inst,&n,devs);
  const char* kinds[]={"OTHER","INTEGRATED_GPU","DISCRETE_GPU","VIRTUAL_GPU","CPU"};
  for(uint32_t i=0;i<n;i++){
    unsigned char buf[4096]; memset(buf,0,sizeof buf);
    vkGetPhysicalDeviceProperties(devs[i],buf);
    uint32_t type=*(uint32_t*)(buf+16);
    printf("  [%u] type=%-15s name=%s\n", i, type<5?kinds[type]:"?", (char*)(buf+20));
  }
  return 0;
}
