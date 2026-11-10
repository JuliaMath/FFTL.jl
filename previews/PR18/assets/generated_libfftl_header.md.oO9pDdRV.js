import{_ as a,o as i,c as n,ao as e}from"./chunks/framework.CsA4OsPf.js";const c=JSON.parse('{"title":"C header (fftl.h)","description":"","frontmatter":{},"headers":[],"relativePath":"generated/libfftl_header.md","filePath":"generated/libfftl_header.md","lastUpdated":null}'),l={name:"generated/libfftl_header.md"};function t(h,s,p,r,k,d){return i(),n("div",null,[...s[0]||(s[0]=[e(`<h1 id="C-header-fftl.h" tabindex="-1">C header (fftl.h) <a class="header-anchor" href="#C-header-fftl.h" aria-label="Permalink to &quot;C header (fftl.h) {#C-header-fftl.h}&quot;">​</a></h1><div class="language-c vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">c</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">/* libfftl: FFTL.jl as a C library.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> *</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * The transforms are FFTW3&#39;s C interface: include fftw3.h (next to this file), link with</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * -lfftl, or with -lfftw3 / -lfftw3f, which are links to the same library. fftw3.h lists what</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * differs from FFTW. This header declares what FFTW has no counterpart for: the message of</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * the last failed call, the library version, and the rules of the worker pool baked into the</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * library (Threading, below).</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> */</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#ifndef</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> FFTL_H</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#define</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> FFTL_H</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#include</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &lt;stddef.h&gt;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#ifdef</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> __cplusplus</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">extern</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;C&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#endif</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">/* size_t fftl_last_error(char *buf, size_t buf_len)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> *</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * Writes the message of the last failed call of this thread as a zero-terminated string into</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * buf (at most buf_len bytes including the terminator) and returns the full length of the</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * message, like snprintf. An empty message means no error so far. With buf == NULL it only</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * returns the length. Successful calls do not clear the message.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> *</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * Every NULL a planner of fftw3.h returns leaves its reason here, as do execute functions</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * called with an invalid plan or with arrays the plan cannot take (FFTW returns nothing from</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * them), fftw_destroy_plan and fftw_copy_plan of an invalid plan, and failed allocations.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> */</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">size_t</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> fftl_last_error</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">char</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">buf</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">size_t</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;"> buf_len</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">/* size_t fftl_version(char *buf, size_t buf_len)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> *</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * Writes the library version as a zero-terminated &quot;major.minor.patch&quot; string into buf (at</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * most buf_len bytes including the terminator) and returns the full length of the version</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * string, like snprintf. With buf == NULL it only returns the length.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> */</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">size_t</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> fftl_version</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">char</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">buf</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">size_t</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;"> buf_len</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">/* ---------------------------------------------------------------------------</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * Threading</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> *</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * libfftl embeds the Julia runtime, whose worker pool is fixed when the library is BUILT and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * cannot change afterwards: build.sh bakes the number through juliac&#39;s --jl-option threads</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * (its FFTL_THREADS variable: a number, or &quot;auto&quot; for one worker per CPU of the machine the</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * library is LOADED on, counted at load time from the process&#39;s CPU affinity; the default is</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * &quot;auto&quot;, on macOS the number of performance cores of the build machine). JULIA_NUM_THREADS</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * in the environment of the loading process has no effect: the baked option takes precedence.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * The Julia runtime cannot grow its pool after it starts, so changing the count needs a</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * rebuild.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> *</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * fftw_plan_with_nthreads sets how many threads later plans use. A count above the baked</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * worker count is allowed and correct: execution queues that plan&#39;s tasks on the available</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * workers, which gives no speedup - and neither does a baked count above the number of cores</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * the process may use. The thread that loads the library is one of the workers: a runtime</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * baked with 4 runs a plan with 4 threads on all 4, while the calling thread waits for them</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * inside the execute function. Small transforms run on one thread regardless.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> *</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * All entry points may be called from any thread of the host program, including threads it</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * created itself, and from several threads at once (the Julia runtime adopts a foreign</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * thread on its first call). Plans are created and destroyed under the library&#39;s registry</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * lock. One plan may execute on several threads at once through the new-array execute</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * functions, as in FFTW: each execution takes its own copy of the plan&#39;s work buffers.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * Destroying a plan while it executes is not allowed.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> * ------------------------------------------------------------------------- */</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#ifdef</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> __cplusplus</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#endif</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#endif</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> /* FFTL_H */</span></span></code></pre></div>`,2)])])}const f=a(l,[["render",t]]);export{c as __pageData,f as default};
