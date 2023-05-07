#include <iostream>
#include <mpi.h>
#include <ctime>
#include <unistd.h> // Per eventuale usleep. Nota: UNIX-Only!
#include <cstdlib>
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_ttf.h>
using namespace std;

const int ROWS = 180;
const int COLS = 280;
const int NUM_GENERAZIONI = 10000;
const int CELLE_INIZIALI = ROWS*120; // Numero arbitrario scelto da me
const int SEND_DOWN_TAG = 70;
const int SEND_UP_TAG = 35;

/********** DICHIARAZIONE STRUCT E PROTOTIPI FUNZIONI **********/

struct PorzioneSlave 
{
    // Ogni processo ha una mini matrice principale e di supporto (array 1D)
    bool* miaMatPrincipaleLineare;
    bool* miaMatSupportoLineare;
    // Ogni processo ha due righe ghost
    bool rigaGhostSopra[COLS] = {false};
    bool rigaGhostSotto[COLS] = {false};   
    // Ogni processo ha un rank
    int my_rank;
    
    // Facilitatori d'accesso
    bool** miaMatPrincipale;
    bool** miaMatSupporto;

    // Ogni processo ha un certo numero di righe
    int my_rows_number;

    PorzioneSlave(int rows_per_proc, int rank, int num_procs) : my_rank(rank) 
    {
        my_rows_number = rows_per_proc;
        int dimMatLineare = rows_per_proc*COLS;

        // In base alla divisibilità e al rank, capisco quante righe assegnare al processo
        if (my_rank == num_procs-1 && ROWS%num_procs != 0) {
            my_rows_number = rows_per_proc+(ROWS%num_procs);
            dimMatLineare = ((rows_per_proc*COLS)) + ((ROWS%num_procs)*COLS);
        }

        // Costruisco la mia matrice
        miaMatPrincipaleLineare = new bool[dimMatLineare];
        miaMatSupportoLineare = new bool[dimMatLineare];

        // Costruisco i facilitatori d'accesso
        miaMatPrincipale = new bool*[my_rows_number];
        miaMatSupporto = new bool*[my_rows_number];
        for (int i = 0; i < my_rows_number; i++) {
            miaMatPrincipale[i] = miaMatPrincipaleLineare+COLS*i;
            miaMatSupporto[i] = miaMatSupportoLineare+COLS*i;
        }
    }

    ~PorzioneSlave() {
        delete[] miaMatPrincipaleLineare;
        delete[] miaMatPrincipale;
        delete[] miaMatSupportoLineare;
        delete[] miaMatSupporto;
    }

    void swapMatrici() 
    {
        for (int i = 0; i < my_rows_number; i++) {  
            miaMatPrincipale[i] = miaMatSupportoLineare+COLS*i;
            miaMatSupporto[i] = miaMatPrincipaleLineare+COLS*i;
        }
        bool* temp = miaMatPrincipaleLineare;
        miaMatPrincipaleLineare = miaMatSupportoLineare;
        miaMatSupportoLineare = temp;
    }
};

void aggiornaContVicini(PorzioneSlave &, int , int &, int );
void aggiornaMatrice(PorzioneSlave &, int , int , int );
void stampaMatrice(bool* , ALLEGRO_FONT *, int);
void aggiornaRigheGhost(PorzioneSlave &, int, MPI_Datatype );

int main(int argc, char** argv)
{
    /********** INIZIALIZZAZIONE LIBRERIA E RISORSE **********/
    
    srand(time(0));
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    bool *matPrincipale;
    bool *matSupporto;
    int rows_per_proc = ROWS/num_procs;
    ALLEGRO_DISPLAY *display = nullptr;
    ALLEGRO_FONT *font = nullptr;

    MPI_Datatype rowtype;
    MPI_Type_contiguous(COLS, MPI_CXX_BOOL, &rowtype);
    MPI_Type_commit(&rowtype);

    // Alloco spazio per due send, perchè magari la prima non ha finito e parte la seconda.
    int bufferSize = 2*(COLS+MPI_BSEND_OVERHEAD);
    bool buffer[bufferSize];
    MPI_Buffer_attach(&buffer, bufferSize);

    double time;

    if (my_rank == 0) 
    {
        time = MPI_Wtime();
        if (!al_init()) {
            cerr << "failed to initialize allegro!" << endl;
            return -1;
        }

        // Faccio partire Allegro
        ALLEGRO_DISPLAY_MODE disp_data;
        al_set_new_display_flags(ALLEGRO_FULLSCREEN_WINDOW);
        al_get_display_mode(0, &disp_data);
        // RIMUOVI PER STAMPA
        display = al_create_display(disp_data.width, disp_data.height);
        al_init_font_addon();
        al_init_ttf_addon();
        al_init_primitives_addon();
        font = al_load_ttf_font("opensans.ttf", 16, 0);

        // Creo le matrici del master e le popolo
        matPrincipale = new bool[ROWS*COLS]();
        matSupporto = new bool[ROWS*COLS]();

        for (int k = 0; k < CELLE_INIZIALI; k++)
            matPrincipale[rand()%(ROWS*COLS)] = true;
        // RIMUOVI PER STAMPA
        stampaMatrice(matPrincipale, font, 0);
    }

    /********** RIPARTIZIONE DELLE STRUTTURE DATI TRA I VARI PROCESSI **********/

    PorzioneSlave miaPorzione(rows_per_proc, my_rank, num_procs);
    
    int* displacements = new int[num_procs];
    int* sendCounts = new int[num_procs];

    for (int i = 0; i < num_procs; i++) 
        displacements[i] = i*((rows_per_proc*COLS));

    for (int i = 0; i < num_procs; i++) {
        if (i == num_procs-1 && ROWS%num_procs != 0)
            sendCounts[i] = ((rows_per_proc*COLS)) + ((ROWS%num_procs)*COLS);
        else
            sendCounts[i] = ((rows_per_proc*COLS));
    }

    MPI_Scatterv(matPrincipale, sendCounts, displacements, MPI_CXX_BOOL, miaPorzione.miaMatPrincipaleLineare, sendCounts[my_rank], MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Scatterv(matSupporto, sendCounts, displacements, MPI_CXX_BOOL, miaPorzione.miaMatSupportoLineare, sendCounts[my_rank], MPI_CXX_BOOL, 0, MPI_COMM_WORLD);   
    aggiornaRigheGhost(miaPorzione, num_procs, rowtype);

    /********** LOOP PRINCIPALE DEL GIOCO **********/

    for (int k = 0; k < NUM_GENERAZIONI; k++) 
    {
        for (int i = 0; i < miaPorzione.my_rows_number; i++) 
            for (int j = 0; j < COLS; j++) 
                aggiornaMatrice(miaPorzione, i,j, num_procs);

        miaPorzione.swapMatrici();
        // RIMUOVI PER STAMPA
        MPI_Gatherv(miaPorzione.miaMatPrincipaleLineare, sendCounts[my_rank], MPI_CXX_BOOL, matPrincipale, sendCounts, displacements, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        MPI_Gatherv(miaPorzione.miaMatSupportoLineare, sendCounts[my_rank], MPI_CXX_BOOL, matSupporto, sendCounts, displacements, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        
        if (my_rank == 0) 
            stampaMatrice(matPrincipale, font, k);   
        aggiornaRigheGhost(miaPorzione, num_procs, rowtype);
    }

    MPI_Buffer_detach(&buffer, &bufferSize);
    MPI_Type_free(&rowtype);
    if (my_rank == 0)
        printf("Tempo impiegato con %d processi: %f\n", num_procs, MPI_Wtime()-time);

    MPI_Finalize();
    return 0;
}

void aggiornaContVicini(PorzioneSlave &miaPorzione, int x, int y, int &contVicini, int num_procs)
{
    if (x < 0) {
        if (miaPorzione.rigaGhostSopra[y]) 
            contVicini++;
    }
    else if (x >= miaPorzione.my_rows_number) {
        if (miaPorzione.rigaGhostSotto[y]) 
            contVicini++;
    }
    else {
        if (miaPorzione.miaMatPrincipale[x][y]) 
            contVicini++;   
    }       
}

void aggiornaMatrice(PorzioneSlave &miaPorzione, int i, int j, int num_procs) 
{
    int contVicini = 0;

    if (j-1 >= 0) 
        aggiornaContVicini(miaPorzione, i-1, j-1, contVicini, num_procs);
    else 
        aggiornaContVicini(miaPorzione, i-1, COLS-1, contVicini, num_procs);
    
    aggiornaContVicini(miaPorzione, i-1, j, contVicini, num_procs);

    if (j+1 < COLS)
        aggiornaContVicini(miaPorzione, i-1, j+1,contVicini, num_procs);
    else
        aggiornaContVicini(miaPorzione, i-1, 0,contVicini, num_procs);

    if (j-1 >= 0) 
        aggiornaContVicini(miaPorzione, i+1, j-1, contVicini, num_procs);
    else
        aggiornaContVicini(miaPorzione, i+1, COLS-1, contVicini, num_procs);
    
    aggiornaContVicini(miaPorzione, i+1, j,contVicini, num_procs);

    if (j+1 < COLS) 
        aggiornaContVicini(miaPorzione, i+1, j+1, contVicini, num_procs);
    else 
        aggiornaContVicini(miaPorzione, i+1, 0, contVicini, num_procs);


    /* Guardo sulla mia stessa riga */
    if ((j-1 >= 0 && miaPorzione.miaMatPrincipale[i][j-1]) || (j-1 < 0 && miaPorzione.miaMatPrincipale[i][COLS-1])) 
        contVicini++;
    if ((j+1 < COLS && miaPorzione.miaMatPrincipale[i][j+1]) || (j+1 >= COLS && miaPorzione.miaMatPrincipale[i][0])) 
        contVicini++;
    
    /* Tiro le somme: la cella nasce, rimane viva, o muore. */
    if (miaPorzione.miaMatPrincipale[i][j]) {
        if (contVicini > 3 || contVicini < 2) 
            miaPorzione.miaMatSupporto[i][j] = false;
        else
            miaPorzione.miaMatSupporto[i][j] = true;
    }
    else {
        if (contVicini == 3)
            miaPorzione.miaMatSupporto[i][j] = true;
        else
            miaPorzione.miaMatSupporto[i][j] = false;
    }
}

void stampaMatrice(bool* mat, ALLEGRO_FONT *font, int k) 
{
    al_clear_to_color(al_map_rgb(200, 200, 200));
    
    int i = 0, j = 0;
    for (int k = 0; k < ROWS*COLS; k++) {
        if (mat[k])
            al_draw_filled_rectangle(j * 4.5, i * 4.4, j * 4.5 + 4.5, i * 4.4 + 4.4, al_map_rgb(50, 205, 50));

        j++;
        if (j == COLS) {
            j = 0;
            i++;
        }
    }

    string s = "Generazione " + to_string(k);
    al_draw_text(font, al_map_rgb(0,0,0), 20, 20, 0, s.c_str());
    al_flip_display();
    //usleep(500000); 
}

void aggiornaRigheGhost(PorzioneSlave &miaPorzione, int num_procs, MPI_Datatype rowtype)
{
    MPI_Status status;
    MPI_Request request;

    // CASO LIMITE: C'è un solo processo
    if (num_procs == 1) {
        for (int i = 0; i < COLS; i++) {
            miaPorzione.rigaGhostSopra[i] = miaPorzione.miaMatPrincipale[miaPorzione.my_rows_number-1][i];
            miaPorzione.rigaGhostSotto[i] = miaPorzione.miaMatPrincipale[0][i];
        }
        return;
    }

    if (miaPorzione.my_rank != 0 && miaPorzione.my_rank != num_procs-1) {
        MPI_Ibsend(miaPorzione.miaMatPrincipale[0], 1, rowtype, miaPorzione.my_rank-1, SEND_UP_TAG, MPI_COMM_WORLD, &request);
        MPI_Ibsend(miaPorzione.miaMatPrincipale[miaPorzione.my_rows_number-1], 1, rowtype, miaPorzione.my_rank+1, SEND_DOWN_TAG, MPI_COMM_WORLD, &request);
    }
    else if (miaPorzione.my_rank == 0) {
        MPI_Ibsend(miaPorzione.miaMatPrincipale[0], 1, rowtype, num_procs-1, SEND_UP_TAG, MPI_COMM_WORLD, &request);
        MPI_Ibsend(miaPorzione.miaMatPrincipale[miaPorzione.my_rows_number-1], 1, rowtype, miaPorzione.my_rank+1, SEND_DOWN_TAG, MPI_COMM_WORLD, &request);
    }
    else if (miaPorzione.my_rank == num_procs-1) {
        MPI_Ibsend(miaPorzione.miaMatPrincipale[0], 1, rowtype, miaPorzione.my_rank-1, SEND_UP_TAG, MPI_COMM_WORLD, &request);
        MPI_Ibsend(miaPorzione.miaMatPrincipale[miaPorzione.my_rows_number-1], 1, rowtype, 0, SEND_DOWN_TAG, MPI_COMM_WORLD, &request);
    }

    if (miaPorzione.my_rank != 0 && miaPorzione.my_rank != num_procs-1) {
        MPI_Recv(miaPorzione.rigaGhostSopra, 1, rowtype, miaPorzione.my_rank-1, SEND_DOWN_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(miaPorzione.rigaGhostSotto, 1, rowtype, miaPorzione.my_rank+1, SEND_UP_TAG, MPI_COMM_WORLD, &status);
    }
    else if (miaPorzione.my_rank == 0) {
        MPI_Recv(miaPorzione.rigaGhostSopra, 1, rowtype, num_procs-1, SEND_DOWN_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(miaPorzione.rigaGhostSotto, 1, rowtype, miaPorzione.my_rank+1, SEND_UP_TAG, MPI_COMM_WORLD, &status);
    }
    else if (miaPorzione.my_rank == num_procs-1) {
        MPI_Recv(miaPorzione.rigaGhostSopra, 1, rowtype, miaPorzione.my_rank-1, SEND_DOWN_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(miaPorzione.rigaGhostSotto, 1, rowtype, 0, SEND_UP_TAG, MPI_COMM_WORLD, &status);
    }
}
