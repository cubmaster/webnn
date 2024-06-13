//https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/chat/main.js#L81
import { Component,OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LlmService } from './services/llm.service';
import { env, AutoTokenizer } from '@xenova/transformers';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet,FormsModule,CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit{
  MODELS = [
    { name: "phi3", path: "microsoft/Phi-3-mini-4k-instruct-onnx-web", externaldata: true },
    { name: "phi3dev", path: "schmuell/Phi-3-mini-4k-instruct-onnx-web", externaldata: true }
  ]
  selectedModel:string="phi3";
  result:string="no results yet";
  Ready=false;
  config = {
    provider: "webgpu",
    profiler: 0,
    verbose: 0,
    threads: 1,
    show_special: 0,
    csv: 0,
    max_tokens: 200,
    local: 0,
  }
  userInput="Tell me a story about ships";
  history:string[] = [];
  title = 'webnn';
  tokenizer:any;

  constructor(public llm:LlmService){

  }
  async ngOnInit(){

    await this.Init();
    this.llm.Status.subscribe(this.handleStatus);
  }

  async Init() {
    const currentModel = this.MODELS.find((model) => model.name === this.selectedModel)|| this.MODELS[0];
    //this.tokenizer = await AutoTokenizer.from_pretrained(currentModel.path);

    await this.llm.load(currentModel, {
      provider: this.config.provider,
      profiler: this.config.profiler,
      verbose: this.config.verbose,
      local: this.config.local,
      max_tokens: this.config.max_tokens,
      hasFP16: this.llm.hasFP16, 
      show_special: this.config.show_special,
    });
    this.Ready=true;



  }

  handleStatus(status) {
 
    //this.result = status + "\n";
  }

  async submitRequest() {
    this.Ready=false;
    this.history.push(this.userInput);
    this.result = await this.llm.query(this.userInput,true);
    this.Ready=true;
  }

}
