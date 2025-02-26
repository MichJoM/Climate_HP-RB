import openai
import os
import json
import argparse
import yaml
from tqdm import tqdm
import time
import csv
from typing import Dict, List, Optional


class YouTubePropagandaInference:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.setup_openai()
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def setup_openai(self) -> None:
        """Setup OpenAI credentials with error checking"""
        try:
            self.client = openai.OpenAI(
                organization=os.environ["OPENAI_ORGANIZATION"],
                api_key=os.getenv("OPENAI_API_KEY")
            )
            if not self.client.api_key:
                raise ValueError("OpenAI API key not found in environment variables")
        except Exception as e:
            print(f"Error setting up OpenAI credentials: {str(e)}")
            raise

    def load_config(self) -> None:
        """Load configuration with error checking"""
        try:
            with open(self.config_path, 'r') as stream:
                self.model_config = yaml.safe_load(stream)
            required_fields = ['model_name', 'instruction', 'prompt_type',
                             'input_data_path', 'output_path']
            for field in required_fields:
                if field not in self.model_config:
                    raise ValueError(f"Missing required field in config: {field}")
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise

    def prompt_gen(self, input_text: str) -> str:
        """Generate prompt with the same propaganda techniques definitions"""
        prompt_instruction = """
Sei un esperto nell'analizzare testi persuasivi e identificare tecniche di persuasione e manipolazione, incluse quelle implicite. Analizza attentamente ogni testo fornito, considerando sia il significato letterale che quello implicito. Di seguito le tecniche retoriche.

Loaded_Language :
Utilizzo di parole e frasi specifiche con forti implicazioni emotive (sia positive che negative) per influenzare e convincere il pubblico. Possono essere utilizzate parole scurrili. L'essenza di questa tecnica è l'uso di termini che vanno oltre il loro significato letterale per evocare una risposta emotiva. 

Exaggeration_Minimisation :
Rappresentare qualcosa in modo eccessivo: rendere le cose più grandi, migliori, peggiori (es. "il migliore dei migliori", "qualità garantita") o far sembrare qualcosa meno importante o più piccolo di quanto sia in realtà (es. dire che un insulto era solo uno scherzo), minimizzando dichiarazioni e ignorando argomenti e accuse fatte da un avversario.

Slogan/Conversation_Killer :
Frasi brevi e incisive per scoraggiare il pensiero critico e/o esortare a compiere una certa azione attraverso un'apparente definitività del messaggio. Spesso si richiamano alla saggezza popolare, apparentemente incontestabile, o a stereotipi per evitare ulteriori discussioni.

Appeal_to_Time :
Argomento centrato sull'idea che sia giunto il momento di una particolare azione, oppure che non ci sia più tempo da perdere. L'appello ad "Agire Ora!".

Appeal_to_Values/Flag_Waving :
Fa leva su valori identitari (nazionalismo, patriottismo, appartenenza a un gruppo/ceto sociale) morali e sociali considerati positivi dal pubblico target (libertà, democrazia, etica, religione) per promuovere o giustifica un'idea. Si basa sul presupposto che i destinatari abbiano già determinati pregiudizi o convinzioni. 

Appeal_to_Authority :
Quando per sostenere o giustificare una tesi, si cita una autorità come fonte, che può essere o meno effettivamente competente nel campo. 

Appeal_to_Popularity :
Giustificare un'idea sostenendo che "tutti" sono d'accordo o che "nessuno" è in disaccordo, incoraggiando il pubblico ad adottare la stessa posizione per conformismo. "Tutti" può riferirsi al pubblico generale, esperti (tutti gli esperti dicono che...), paesi o altri gruppi.

Appeal_to_Fear :
Promuovere o respingere un'idea sfruttando la repulsione o la paura del pubblico, descrivendo possibili scenari in modo spaventoso (terribili cose che potrebbero succedere) per instillare paura.

Straw_Man/Red_Herring :
La discussione viene distolta dall'argomento originale attraverso l’introduzione di argomenti apparentemente coerenti, ma diversi al tema principale. Così si sposta l’attenzione su un tema secondario.

Tu_Quoque/Whataboutism :
Si scredita una posizione o un avversario evidenziando presunte contraddizioni o doppi standard. Può manifestarsi evidenziando incoerenze sullo stesso tema o introducendo comparazioni con altri ambiti o situazioni. L'obiettivo è minare la credibilità attraverso paragoni con altre questioni.

Repetition :
Uso ripetuto della stessa parola, frase, storia o immagine nella speranza che la ripetizione porti a persuadere il pubblico.

Intentional_Confusion_Vagueness :
Uso di parole deliberatamente poco chiare in modo che il pubblico possa avere le proprie interpretazioni. Ad esempio, quando nell'argomentazione viene utilizzata una frase poco chiara con definizioni multiple o poco chiare e, quindi, non supporta la conclusione.

Name_Calling :
Quando dei nomi o aggettivi sono dati ad un individuo, istituzione o gruppo con intento denigratorio o per metterne in discussione l’autorità. Riguarda specificamente la caratterizzazione del soggetto attraverso aggettivi, sostantivi o riferimenti a orientamenti politici, opinioni, caratteristiche personali o appartenenze organizzative.

Reductio_ad_Hitlerum :
Attaccare un avversario o un'attività associandoli ad un altro gruppo, attività o concetto che ha forti connotazioni negative per il pubblico target. La tecnica opera stabilendo un collegamento o un'equivalenza tra il bersaglio e qualsiasi individuo, gruppo o evento (presente o passato) che ha una percezione indiscutibilmente negativa o viene presentato come tale. L'obiettivo è trasferire la negatività dell'associazione al soggetto criticato.

Smear/Doubt : 
Tecnica che mira a minare la credibilità di qualcuno o qualcosa (ad esempio enti/istituzioni) questionando specifiche competenze o capacità, attaccando la reputazione e il carattere morale complessivo, mettendo in dubbio le intenzioni alla base di una scelta. 

Causal_Oversimplification/Consequential_Oversimplification
Tecnica usata per ridurre un fenomeno complesso ad una singola causa, ignorando altri fattori, spesso per supportare una narrativa o soluzione specifica (secondo la logica Y è successo dopo X, quindi X è la causa di Y", oppure "X ha causato Y, quindi X è l'unica causa di Y). Usata anche per affermare che un certo evento/azione porterà a una catena di eventi a effetto domino con conseguenze negative (per respingere l'idea) o positive (per supportarla). In questo caso assume la forma di : se succederà A, allora B, C, D succederanno. 

False_Dilemma_No_Choice :
Presentare una situazione come se avesse solo due alternative quando in realtà esistono più opzioni. Nella sua forma estrema, presenta una sola possibile linea d'azione, eliminando tutte le altre scelte. L'essenza principale della False_Dilemma è limitare artificialmente la gamma di possibili soluzioni o punti di vista, spesso per forzare una particolare conclusione o corso d'azione.
        """

        prompt_base = """
        Stai performando un multilabel detection task. Analizza il seguente testo molto attentamente e individua l'eventuale presenza di una o più delle tecniche di persuasione sopra definite.
        Considera che:
        - Le tecniche possono sovrapporsi: la stessa frase può utilizzare più tecniche contemporaneamente. Se ne usa più di una non puoi inserire no_technique_detected.
        - Le tecniche possono essere espresse in modo sarcastico o indiretto
        - Il tono e il contesto sono importanti quanto le parole specifiche
        - Una tecnica può manifestarsi attraverso una serie di affermazioni correlate, non necessariamente in una singola frase
        - Non necessariamente il testo contiene una tecnica, però è molto importante che lo analizzi a fondo per evitare ogni dubbio 
        
        Se nessuna tecnica viene rilevata, rispondi "no_technique_detected". Altrimenti, elenca le tecniche individuate, usando lo stesso formato in cui sono definite sopra.
        Rispondi solo e solamente con l'elenco delle tecniche (o no_technique_detected se non ce ne sono), altri formati non verranno accettati.
        Di seguito, ripropongo le etichette precise da usare nell'output:

        Slogan/Conversation_Killer
        Appeal_to_Time
        Appeal_to_Values/Flag_Waving
        Appeal_to_Authority
        Appeal_to_Popularity
        Appeal_to_Fear
        Straw_Man/Red_Herring
        Tu_Quoque/Whataboutism
        Loaded_Language
        Repetition
        Intentional_Confusion_Vagueness
        Exaggeration_Minimisation
        Name_Calling
        Reductio_ad_Hitlerum
        Smear/Doubt
        Causal_Oversimplification/Consequential_Oversimplification
        False_Dilemma_No_Choice
        no_technique_detected
        
        Ecco il testo da analizzare:"""


        return f'{prompt_instruction} {prompt_base} <{input_text}>'

    def inference(self, prompt: str) -> str:
        """Make API call with robust error handling and retries"""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_config['model_name'],
                    messages=[
                        {"role": "system", "content": prompt.split("Ecco il testo da analizzare:")[0].strip()},
                        {"role": "user", "content": prompt.split("Ecco il testo da analizzare:")[1].strip()}
                    ],
                    max_tokens=1000,
                    temperature=0.2,
                )

                return completion.choices[0].message.content

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.error_count += 1
                    print(f"All retries failed. Total errors: {self.error_count}")
                    return "no technique detected"

    def process_output(self, output: str) -> List[str]:
        """Process model output with validation"""
        if not output or output.lower().strip() == 'no technique detected':
            return []

        valid_techniques = {
            'Slogan/Conversation_Killer',
            'Appeal_to_Time'
            'Appeal_to_Authority',
            'Appeal_to_Popularity',
            'Appeal_to_Values/Flag_Waving',
            'Appeal_to_Fear',
            'Straw_Man/Red_Herring',
            'Loaded_Language',
            'Repetition',
            'Intentional_Confusion_Vagueness',
            'Exaggeration_Minimisation',
            'Name_Calling',
            'Reductio_ad_Hitlerum',
            'Tu_Quoque/Whataboutism',
            'Smear/Doubt',
            'Causal_Oversimplification/Consequential_Oversimplification',
            'False_Dilemma_No_Choice',
            'no_technique_detected'
        }

        techniques = []
        for line in output.split('\n'):
            technique = line.strip()
            if technique in valid_techniques:
                techniques.append(technique)

        return techniques

    def save_results(self) -> None:
        """Process CSV and save results with error handling"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.model_config['output_path']), exist_ok=True)

            # Read and validate input CSV
            rows = []
            try:
                with open(self.model_config['input_data_path'], 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    rows = list(csv_reader)
            except Exception as e:
                print(f"Error reading input CSV file: {str(e)}")
                raise

            print("Processing CSV rows...")

            # Prepare CSV writer for output
            fieldnames = ['article_id', 'paragraph', 'text', 'techniques']
            with open(self.model_config['output_path'], 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for row in tqdm(rows):
                    try:
                        article_id = row.get('article_id')
                        paragraph = row.get('paragraph')
                        text = row.get('text', '').strip()

                        if not text:
                            continue

                        prompt = self.prompt_gen(text)
                        output = self.inference(prompt)
                        techniques = self.process_output(output)

                        result = {
                            'article_id': article_id,
                            'paragraph': paragraph,
                            'text': text,
                            'techniques': ','.join(techniques) if techniques else 'no technique detected'
                        }

                        writer.writerow(result)

                    except Exception as e:
                        print(f"Error processing row with article_id {article_id}, paragraph {paragraph}: {str(e)}")
                        continue

            print(f"Results saved to {self.model_config['output_path']}")
            if self.error_count > 0:
                print(f"Total API errors encountered: {self.error_count}")

        except Exception as e:
            print(f"Critical error in save_results: {str(e)}")
            raise

    def run_all(self) -> None:
        """Main execution method with error handling"""
        try:
            self.save_results()
        except Exception as e:
            print(f"Failed to complete execution: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()

    try:
        inference = YouTubePropagandaInference(args.config_path)
        inference.run_all()
    except Exception as e:
        print(f"Program failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()