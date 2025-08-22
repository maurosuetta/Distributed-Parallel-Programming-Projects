#include <stdio.h>

#include <assert.h>
#include <mpi.h>

#include "auxiliar.h"

/// TODO
/// Reading the planes from a file for MPI
void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    int num_planes = 0;

    // Read header
    fgets(line, sizeof(line), file);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Map: %lf, %lf : %d %d", &(*x_max), &(*y_max), N, M);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Number of Planes: %d", &num_planes);
    fgets(line, sizeof(line), file);

    // Compute tiles per proc
    int total_cells = (*N) * (*M);
    for (int i = 0; i <= size; i++)
        tile_displacements[i] = (i * total_cells) / size;
    //if (rank == 0)
    //    for (int i = 0; i < size; i++)
    //        printf("%d: %d %d : %d\n", i, tile_displacements[i], tile_displacements[i+1]-1, tile_displacements[i+1]-tile_displacements[i]);

    // Read grid data
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        int idx;
        double x, y, vx, vy;
        if (sscanf(line, "%d %lf %lf %lf %lf",
                   &idx, &x, &y, &vx, &vy) == 5) {
            int index_i = get_index_i(x, *x_max, *N);
            int index_j = get_index_j(y, *y_max, *M);
            int index_map = get_index(index_i, index_j, *N, *M);
            if (index_map >= tile_displacements[rank] && index_map < tile_displacements[rank+1])
                insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
            index++;
        }
    }
    fclose(file);
    int total_planes_recv = 0;
    MPI_Allreduce(&total_planes_recv, &num_planes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Total planes read: %d\n", index);
    assert(total_planes_recv == num_planes);
}

/// TODO
/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    PlaneNode* current = list->head;

    int planes_to_send[size];
    int planes_to_recv[size];
    for (int i = 0; i < size; i++)
        planes_to_send[i] = 0;

    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int tp1_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        if (tp1_rank != rank) {
            planes_to_send[tp1_rank] += 1;
            //printf("It will send %d from %d to %d\n", current->index_plane, rank, tp1_rank);
        }
        current = current->next;
    }

    MPI_Alltoall(planes_to_send, 1, MPI_INT, planes_to_recv, 1, MPI_INT, MPI_COMM_WORLD);

    int total_planes_to_send = 0;
    for (int i = 0; i < size; i++)
        total_planes_to_send += planes_to_send[i];
    int total_planes_to_recv = 0;
    for (int i = 0; i < size; i++)
        total_planes_to_recv += planes_to_recv[i];

    MPI_Request requests[total_planes_to_send];

    int index_data = 0;
    int index_data_rank[size];
    for (int i = 0; i < size; i++)
        index_data_rank[i] = 0;

    double* data_to_send[size];
    for (int i = 0; i < size; i++)
        data_to_send[i] = (double*)malloc(5 * planes_to_send[i] * sizeof(double));

    current = list->head;
    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int tp1_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        PlaneNode* next = current->next;
        if (tp1_rank != rank) {
            double* data_to_send_i = &(data_to_send[tp1_rank][5*index_data_rank[tp1_rank]]);
            data_to_send_i[0] = (double)current->index_plane;
            data_to_send_i[1] = current->x;
            data_to_send_i[2] = current->y;
            data_to_send_i[3] = current->vx;
            data_to_send_i[4] = current->vy;

            MPI_Isend(data_to_send_i, 5, MPI_DOUBLE, tp1_rank, 0, MPI_COMM_WORLD, &requests[index_data]);
            //printf("Sending %d from %d to %d\n", current->index_plane, rank, tp1_rank);
            index_data++;
            index_data_rank[tp1_rank]++;
            remove_plane(list, current);
        }
        current = next;
    }
    assert(index_data == total_planes_to_send);
    for (int i = 0; i < size; i++)
        assert(index_data_rank[i] == planes_to_send[i]);

    MPI_Waitall(total_planes_to_send, requests, MPI_STATUSES_IGNORE);

    index_data = 0;
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < planes_to_recv[i]; k++)
        {
            double data_to_recv[5] = {-1.0, -1.0, -1.0, -1.0, -1.0};
            MPI_Recv(data_to_recv, 5, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("Received %d in %d from %d\n", (int)data_to_recv[0], rank, i);
            //printf("%lf %lf %lf %lf %lf\n", data_to_recv[0], data_to_recv[1], data_to_recv[2], data_to_recv[3], data_to_recv[4]);
            int index = get_index(get_index_i(data_to_recv[1], x_max, N), get_index_j(data_to_recv[2], y_max, M), N, M);
            insert_plane(list, (int)data_to_recv[0], index, rank, data_to_recv[1], data_to_recv[2], data_to_recv[3], data_to_recv[4]);
            index_data++;
        }
    }
}

void communicate_planes_send2(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    PlaneNode* current = list->head;

    int planes_to_send[size];
    int planes_to_recv[size];
    for (int i = 0; i < size; i++)
        planes_to_send[i] = 0;

    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int tp1_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        if (tp1_rank != rank) {
            planes_to_send[tp1_rank] += 1;
            //printf("It will send %d from %d to %d\n", current->index_plane, rank, tp1_rank);
        }
        current = current->next;
    }

    MPI_Alltoall(planes_to_send, 1, MPI_INT, planes_to_recv, 1, MPI_INT, MPI_COMM_WORLD);

    int total_planes_to_send = 0;
    for (int i = 0; i < size; i++)
        total_planes_to_send += planes_to_send[i];
    int total_planes_to_recv = 0;
    for (int i = 0; i < size; i++)
        total_planes_to_recv += planes_to_recv[i];

    MPI_Request requests_send[total_planes_to_send];
    MPI_Request requests_recv[total_planes_to_recv];

    int index_data = 0;
    double* data_to_recv[size];
    for (int i = 0; i < size; i++) {
        data_to_recv[i] = (double*)malloc(5 * planes_to_recv[i] * sizeof(double));
    }
    for (int i = 0; i < size; i++) {
        int index_data_i = 0;
        for (int k = 0; k < planes_to_recv[i]; k++)
        {
            MPI_Irecv(&data_to_recv[i][index_data_i*5], 5, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests_recv[index_data]);
            index_data++;
            index_data_i++;
        }
        assert(index_data_i == planes_to_recv[i]);
    }
    assert(index_data == total_planes_to_recv);

    index_data = 0;
    int index_data_rank[size];
    for (int i = 0; i < size; i++)
        index_data_rank[i] = 0;

    double* data_to_send[size];
    for (int i = 0; i < size; i++)
        data_to_send[i] = (double*)malloc(5 * planes_to_send[i] * sizeof(double));

    current = list->head;
    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int tp1_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        PlaneNode* next = current->next;
        if (tp1_rank != rank) {
            double* data_to_send_i = &(data_to_send[tp1_rank][5*index_data_rank[tp1_rank]]);
            data_to_send_i[0] = (double)current->index_plane;
            data_to_send_i[1] = current->x;
            data_to_send_i[2] = current->y;
            data_to_send_i[3] = current->vx;
            data_to_send_i[4] = current->vy;

            MPI_Isend(data_to_send_i, 5, MPI_DOUBLE, tp1_rank, 0, MPI_COMM_WORLD, &requests_send[index_data]);
            printf("Sending %d from %d to %d\n", current->index_plane, rank, tp1_rank);
            index_data++;
            index_data_rank[tp1_rank]++;
            remove_plane(list, current);
        }
        current = next;
    }
    assert(index_data == total_planes_to_send);

    MPI_Waitall(total_planes_to_send, requests_send, MPI_STATUSES_IGNORE);
    MPI_Waitall(total_planes_to_recv, requests_recv, MPI_STATUSES_IGNORE);

    index_data = 0;
    for (int i = 0; i < size; i++) {
        int index_data_i = 0;
        for (int k = 0; k < planes_to_recv[i]; k++)
        {
            double* data = &data_to_recv[i][index_data_i*5];
            int index = get_index(get_index_i(data[1], x_max, N), get_index_j(data[2], y_max, M), N, M);
            insert_plane(list, (int)data[0], index, rank, data[1], data[2], data[3], data[4]);
            index_data++;
            index_data_i++;
        }
    }

    for (int i = 0; i < size; i++) {
        free(data_to_recv[i]);
        free(data_to_send[i]);
    }

}

/// TODO
/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    PlaneNode* current = list->head;

    int planes_to_send[size];
    int planes_to_recv[size];
    for (int i = 0; i < size; i++)
        planes_to_send[i] = 0;

    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        if (index_i >= N || index_j >= M) {
            printf("Error: x: %.2lf,%.2lf [%.2lf,%.2lf]  -  index %d,%d [%d,%d]\n", current->x, current->y, x_max, y_max, index_i, index_j, N, M);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int tp1_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        if (tp1_rank != rank) {
            planes_to_send[tp1_rank] += 1;
            //printf("It will send %d from %d to %d\n", current->index_plane, rank, tp1_rank);
        }
        current = current->next;
    }

    MPI_Alltoall(planes_to_send, 1, MPI_INT, planes_to_recv, 1, MPI_INT, MPI_COMM_WORLD);

    int index_to_send_first[size+1];
    index_to_send_first[0]=0;
    int total_planes_to_send = 0;
    for (int i = 0; i < size; i++) {
        total_planes_to_send += planes_to_send[i];
        index_to_send_first[i+1] = total_planes_to_send;
    }

    int total_planes_to_recv = 0;
    for (int i = 0; i < size; i++)
        total_planes_to_recv += planes_to_recv[i];

    double* all_data_to_send = (double*)malloc(total_planes_to_send * 5 * sizeof(double));
    int index_data = 0;
    current = list->head;
    int index_to_send[size];
    for (int i = 0; i < size; i++)
        index_to_send[i] = 0;

    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int tp1_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        PlaneNode* next = current->next;
        if (tp1_rank != rank) {
            int index = index_to_send_first[tp1_rank] + index_to_send[tp1_rank];
            all_data_to_send[index*5 + 0] = (double)current->index_plane;
            all_data_to_send[index*5 + 1] = current->x;
            all_data_to_send[index*5 + 2] = current->y;
            all_data_to_send[index*5 + 3] = current->vx;
            all_data_to_send[index*5 + 4] = current->vy;
            //printf("Sending %d from %d to %d\n", current->index_plane, rank, tp1_rank);
            index_data++;
            index_to_send[tp1_rank]++;
            remove_plane(list, current);
        }
        current = next;
    }

    for (int i = 0; i < size; i++)
        assert(index_to_send[i] == planes_to_send[i]);

    int send_count[size], recv_count[size], send_displs[size], recv_displs[size];
    send_displs[0] = 0, recv_displs[0] = 0;
    send_count[0] = planes_to_send[0]*5, recv_count[0] = planes_to_recv[0]*5;
    for (int i = 1; i < size; i++) {
        send_count[i] = planes_to_send[i]*5;
        recv_count[i] = planes_to_recv[i]*5;
        send_displs[i] = send_displs[i - 1] + 5*planes_to_send[i - 1];
        recv_displs[i] = recv_displs[i - 1] + 5*planes_to_recv[i - 1];
    }

    double* all_data_to_recv = (double*)malloc(total_planes_to_recv * 5 * sizeof(double));

    MPI_Alltoallv(all_data_to_send, send_count, send_displs, MPI_DOUBLE, all_data_to_recv, recv_count, recv_displs, MPI_DOUBLE, MPI_COMM_WORLD);

    index_data = 0;
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < planes_to_recv[i]; k++)
        {
            double* data_to_recv = &all_data_to_recv[index_data*5];
            //printf("Received %d in %d from %d\n", (int)data_to_recv[0], rank, i);
            int index = get_index(get_index_i(data_to_recv[1], x_max, N), get_index_j(data_to_recv[2], y_max, M), N, M);
            insert_plane(list, (int)data_to_recv[0], index, rank, data_to_recv[1], data_to_recv[2], data_to_recv[3], data_to_recv[4]);
            index_data++;
        }
    }

    for (int i = 0; i < size; i++) {
        if (send_count[i] != 5 * planes_to_send[i]) {
            printf("Error in send_count[%d]: Expected %d, Found %d\n", i, 5 * planes_to_send[i], send_count[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (recv_count[i] != 5 * planes_to_recv[i]) {
            printf("Error in recv_count[%d]: Expected %d, Found %d\n", i, 5 * planes_to_recv[i], recv_count[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    free(all_data_to_send);
    free(all_data_to_recv);
}

typedef struct {
    int    index_plane;
    double x;
    double y;
    double vx;
    double vy;
} MinPlaneToSend;

/// TODO
/// Communicate planes using all to all calls with custom data types
void communicate_planes_struct_mpi(PlaneList* list,
                               int N, int M,
                               double x_max, double y_max,
                               int rank, int size,
                               int* tile_displacements)
{
    // ------------------------------------------------------------
    // 1) Define and commit the MPI datatype for MPIPlane
    // ------------------------------------------------------------
    MPI_Datatype mpi_plane_type;
    {
        // Our struct has: int index_plane, then four doubles
        const int nitems = 2;                  // 2 “blocks”: one block for int, one for double[4]
        int blocklengths[2]    = {1, 4};
        MPI_Aint offsets[2];
        MPI_Datatype types[2]  = {MPI_INT, MPI_DOUBLE};

        offsets[0] = offsetof(MinPlaneToSend, index_plane);
        offsets[1] = offsetof(MinPlaneToSend, x); // the first double in the struct

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_plane_type);
        MPI_Type_commit(&mpi_plane_type);
    }

    // ------------------------------------------------------------
    // 2) Figure out how many planes each process needs to send
    // ------------------------------------------------------------
    int planes_to_send[size];
    int planes_to_recv[size];
    for (int i = 0; i < size; i++) {
        planes_to_send[i] = 0;
        planes_to_recv[i] = 0;
    }

    PlaneNode* current = list->head;
    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int dest_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        if (dest_rank != rank) {
            planes_to_send[dest_rank]++;
            // Debug (optional):
            // printf("It will send plane %d from %d to %d\n", current->index_plane, rank, dest_rank);
        }
        current = current->next;
    }

    // ------------------------------------------------------------
    // 3) Exchange that information (Alltoall) to know how many planes to receive
    // ------------------------------------------------------------
    MPI_Alltoall(planes_to_send, 1, MPI_INT,
                 planes_to_recv, 1, MPI_INT,
                 MPI_COMM_WORLD);

    // Compute total sends and total receives
    int total_planes_to_send = 0;
    for (int i = 0; i < size; i++)
        total_planes_to_send += planes_to_send[i];

    int total_planes_to_recv = 0;
    for (int i = 0; i < size; i++)
        total_planes_to_recv += planes_to_recv[i];

    // ------------------------------------------------------------
    // 4) Build the array of MPIPlane to send
    // ------------------------------------------------------------
    // Create prefix sums (displacements)
    int send_displs[size];
    int recv_displs[size];
    int send_count[size];
    int recv_count[size];

    // We'll accumulate partial sums to track where each process's block starts
    send_displs[0] = 0;
    recv_displs[0] = 0;
    send_count[0]  = planes_to_send[0];
    recv_count[0]  = planes_to_recv[0];
    for (int i = 1; i < size; i++) {
        send_count[i] = planes_to_send[i];
        recv_count[i] = planes_to_recv[i];
        send_displs[i] = send_displs[i-1] + planes_to_send[i-1];
        recv_displs[i] = recv_displs[i-1] + planes_to_recv[i-1];
    }

    // Allocate space for all planes we need to send
    MinPlaneToSend* all_data_to_send = (MinPlaneToSend*)malloc(total_planes_to_send * sizeof(MinPlaneToSend));
    if (all_data_to_send == NULL) {
        fprintf(stderr, "Error allocating memory for all_data_to_send\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Fill it
    for (int i = 0; i < total_planes_to_send; i++) {
        // We'll fill it in a second pass below
        all_data_to_send[i].index_plane = -1;
        all_data_to_send[i].x = -1.0;
        all_data_to_send[i].y = -1.0;
        all_data_to_send[i].vx = -1.0;
        all_data_to_send[i].vy = -1.0;
    }

    // We'll walk the list again and pack up the planes
    //int* index_to_send = (int*)calloc(size, sizeof(int));
    int index_to_send[size];
    for (int i = 0; i < size; i++)
        index_to_send[i] = 0;

    current = list->head;
    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int dest_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        PlaneNode* next = current->next; // so we can remove and still keep iterating

        if (dest_rank != rank) {
            // The place in the send array for this rank’s block is:
            int idx = send_displs[dest_rank] + index_to_send[dest_rank];
            all_data_to_send[idx].index_plane = current->index_plane;
            all_data_to_send[idx].x           = current->x;
            all_data_to_send[idx].y           = current->y;
            all_data_to_send[idx].vx          = current->vx;
            all_data_to_send[idx].vy          = current->vy;

            index_to_send[dest_rank]++;
            remove_plane(list, current);
        }
        current = next;
    }

    // ------------------------------------------------------------
    // 5) Allocate space for all planes we need to receive
    // ------------------------------------------------------------
    MinPlaneToSend* all_data_to_recv = (MinPlaneToSend*)malloc(total_planes_to_recv * sizeof(MinPlaneToSend));
    if (all_data_to_recv == NULL) {
        fprintf(stderr, "Error allocating memory for all_data_to_recv\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //MinPlaneToSend all_data_to_recv[total_planes_to_recv];

    // ------------------------------------------------------------
    // 6) Exchange the data using MPI_Alltoallv + our custom type
    // ------------------------------------------------------------
    MPI_Alltoallv(
            all_data_to_send,  // send buffer
            send_count,        // how many MPIPlane structs to send to each rank
            send_displs,       // displacement in units of MPIPlane
            mpi_plane_type,    // custom MPI type for the plane

            all_data_to_recv,  // receive buffer
            recv_count,        // how many MPIPlane structs to receive from each rank
            recv_displs,       // displacement in units of MPIPlane
            mpi_plane_type,    // same custom type

            MPI_COMM_WORLD
    );

    // ------------------------------------------------------------
    // 7) Insert the received planes into our list
    // ------------------------------------------------------------
    int offset = 0;
    for (int src_rank = 0; src_rank < size; src_rank++) {
        for (int k = 0; k < planes_to_recv[src_rank]; k++) {
            MinPlaneToSend* p = &all_data_to_recv[offset++];
            // Debug (optional):
            // printf("Rank %d received plane %d from %d\n", rank, p->index_plane, src_rank);

            int i_idx = get_index_i(p->x, x_max, N);
            int j_idx = get_index_j(p->y, y_max, M);
            int index_map = get_index(i_idx, j_idx, N, M);

            insert_plane(list, p->index_plane, index_map, rank, p->x, p->y, p->vx, p->vy);
        }
    }

    // ------------------------------------------------------------
    // 8) Cleanup
    // ------------------------------------------------------------
    free(all_data_to_send);
    free(all_data_to_recv);
    // Done using the MPIPlane type
    MPI_Type_free(&mpi_plane_type);
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    /// TODO
    int rank, size;

    /// TODO
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int tile_displacements[size+1];

    int mode = 0;
    if (argc == 5) {
        input_file = argv[1];
        max_steps = atoi(argv[2]);
        if (max_steps <= 0) {
            fprintf(stderr, "max_steps needs to be a positive integer\n");
            return 1;
        }
        mode = atoi(argv[3]);
        if (mode > 2 || mode < 0) {
            fprintf(stderr, "mode needs to be a value between 0 and 2\n");
            return 1;
        }
        check = atoi(argv[4]);
        if (check >= 2 || check < 0) {
            fprintf(stderr, "check needs to be a 0 or 1\n");
            return 1;
        }
    }
    else {
        fprintf(stderr, "Usage: %s <filename> <max_steps> <mode> <check>\n", argv[0]);
        return 1;
    }

    PlaneList owning_planes = {NULL, NULL};
    read_planes_mpi(input_file, &owning_planes, &N, &M, &x_max, &y_max, rank, size, tile_displacements);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    //print_planes_par_debug(&owning_planes);

    double time_sim = 0., time_comm = 0, time_total=0;

    double start_time = MPI_Wtime();
    int step = 0;
    for (step = 1; step <= max_steps; step++) {
        double start = MPI_Wtime();
        PlaneNode* current = owning_planes.head;
        while (current != NULL) {
            current->x += current->vx;
            current->y += current->vy;
            current = current->next;
        }
        filter_planes(&owning_planes, x_max, y_max);
        time_sim += MPI_Wtime() - start;

        start = MPI_Wtime();
        if (mode == 0)
            communicate_planes_send2(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else if (mode == 1)
            communicate_planes_alltoall(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else if (mode == 2)
            communicate_planes_struct_mpi(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        time_comm += MPI_Wtime() - start;
    }
    time_total = MPI_Wtime() - start_time;

    /// TODO Check computational times
    double times[3] = {time_sim, time_comm, time_total};
    MPI_Allreduce(MPI_IN_PLACE, &times, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check ==1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);

    MPI_Finalize();
    return 0;
}
